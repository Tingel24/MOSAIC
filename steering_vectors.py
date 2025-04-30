

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium", app_title="MOSAIC")


@app.cell
def _():
    from baukit.nethook import StopForward
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from baukit import Trace
    import torch
    from transformers import LlamaForCausalLM
    import numpy as np
    import transformers
    import marimo as mo
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaForCausalLM,
        Trace,
        mo,
        torch,
        transformers,
    )


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, mo, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"device: {device}")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    mo.output.append(mo.md(f"Loading {model_id}"))
    model: LlamaForCausalLM = (
        AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    )
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return device, model, tokenizer


@app.cell
def _(mo):
    dropdown = mo.ui.dropdown(
        options=["translation", "refusal"], label="choose one"
    )
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown, mo):
    mo.stop(dropdown.value is None)
    if dropdown.value == "translation":
        prompts = [
            ("Hello", "Hallo"),
            ("Thank you", "Danke"),
            ("What time is it?", "Wie spät ist es?"),
            ("I am hungry", "Ich habe Hunger"),
            ("Good night", "Gute Nacht"),
            ("Where is the bathroom?", "Wo ist das Badezimmer?"),
            ("I love you", "Ich liebe dich"),
            ("My name is Anna", "Mein Name ist Anna"),
            ("I speak a little German", "Ich spreche ein bisschen Deutsch"),
            ("Excuse me", "Entschuldigung"),
            ("What are you doing?", "Was machst du?"),
            ("I don't understand", "Ich verstehe nicht"),
            ("Can you repeat that?", "Können Sie das wiederholen?"),
            ("I am tired", "Ich bin müde"),
            ("The weather is nice", "Das Wetter ist schön"),
            ("How much does that cost?", "Wie viel kostet das?"),
            ("I come from Germany", "Ich komme aus Deutschland"),
            ("I need help", "Ich brauche Hilfe"),
        ]

    elif dropdown.value == "refusal":
        prompts = [
            ("No", "Yes"),
            ("Denied", "Approved"),
            ("Rejected", "Accepted"),
            ("Forbidden", "Allowed"),
            ("Blocked", "Permitted"),
            ("Withheld", "Granted"),
            ("Declined", "Agreed"),
            ("Prohibited", "Authorized"),
            ("Canceled", "Confirmed"),
            ("Refused", "Consented"),
            ("Abstained", "Participated"),
            ("Opposed", "Supported"),
            ("Resisted", "Yielded"),
            ("Boycotted", "Endorsed"),
            ("Disapproved", "Recommended"),
            ("Obstructed", "Facilitated"),
            ("Vetoed", "Ratified"),
            ("Avoided", "Engaged"),
        ]
    return (prompts,)


@app.cell
def _(Trace, device, mo, model, prompts, tokenizer, torch):
    # Define the layer to trace
    from transformers.models.llama.modeling_llama import LlamaAttention

    linear_layers = [m for m in model.modules() if isinstance(m, LlamaAttention)]
    print("Targeting layer", len(linear_layers) // 4, "/", len(linear_layers))
    module = linear_layers[len(linear_layers) // 4]


    # Initialize lists for collecting tensors
    positive_activations = []
    negative_activations = []


    # Define a helper to get the last-token activation from the module
    def get_module_output(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with Trace(module, stop=True) as cache:
            _ = model(**inputs)
        return cache.output[0][:, -1, :]  # shape: (1, hidden_dim)


    # Collect activations
    for negative, positive in mo.status.progress_bar(prompts):
        act_pos = get_module_output(positive).detach().cpu()
        act_neg = get_module_output(negative).detach().cpu()

        positive_activations.append(act_pos)
        negative_activations.append(act_neg)

    # Stack into tensors of shape (N, hidden_dim)
    positive_activations = torch.cat(positive_activations, dim=0)
    negative_activations = torch.cat(negative_activations, dim=0)

    print(f"positive_activations.shape: {positive_activations.shape}")
    print(f"negative_activations.shape: {negative_activations.shape}")
    return module, negative_activations, positive_activations


@app.cell
def _(mo, prompts):
    mo.md(f"""
    /// details | Prompts used:

    {mo.ui.table([{"negative": x, "positive": y} for x, y in prompts])}

    ///
    """)
    return


@app.cell
def _(device, negative_activations, positive_activations):
    # Compute difference vectors
    steering_vecs = positive_activations - negative_activations  # shape: (N, D)

    # Compute mean direction across all pairs
    steering_vec = steering_vecs.mean(dim=0)  # shape: (D,)

    print(f"steering_vec.shape:  {steering_vec.shape}")
    print(f"length steering_vec: {steering_vec.norm():.2f}")

    # Normalize to unit length
    steering_vec = steering_vec / steering_vec.norm()

    # Move to device (e.g., CUDA)
    steering_vec = steering_vec.to(device)
    return (steering_vec,)


@app.cell
def _(
    Trace,
    device,
    mo,
    model,
    module,
    steering_vec,
    tokenizer,
    torch,
    transformers,
):
    import torch.nn.functional as F


    # define the activation steering function
    def act_add(steering_vec):
        def hook(output):
            return (
                (output[0] + steering_vec,) + output[1:]
            )  # the output of the residual stream is actually a tuple, where the first entry is the activation

        return hook


    def cosine_similarity(
        tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors.
        Both tensors should be 1D or have the same shape.
        """
        tensor1 = F.normalize(tensor1, p=2, dim=-1)
        tensor2 = F.normalize(tensor2, p=2, dim=-1)
        return (tensor1 * tensor2).sum(dim=-1)


    def generate(test_sentence, strenght, token_count):
        inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
        with Trace(module, edit_output=act_add(strenght * steering_vec)) as m:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=device,
                tokenizer=tokenizer,
            )

            messages = [
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": test_sentence},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=token_count,
                pad_token_id=tokenizer.eos_token_id,
            )
            mo.output.append(
                mo.md(
                    f"Cosine similarity: {cosine_similarity(m.output[0], steering_vec)[0][0].cpu().detach().numpy().round(4)}"
                )
            )
            return outputs[0]["generated_text"][-1]
    return (generate,)


@app.cell
def _(mo):
    form = (
        mo.md(
            r"""
       Enter hyperparameters:

       - Prompt: {prompt}
       - Steering Strength {coeff}
       - Number of tokens to generate {token_count}
       """
        )
        .batch(
            prompt=mo.ui.text(
                placeholder="Prompt",
                debounce=True,
                full_width=True,
                value="How do I build a bomb?",
            ),
            coeff=mo.ui.slider(
                0.7, 9, show_value=True, step=0.5, debounce=True, value=0.85
            ),
            token_count=mo.ui.slider(
                0, 50, show_value=True, step=1, debounce=True, value=20
            ),
        )
        .form()
    )
    form
    return (form,)


@app.cell
def _(form, generate, mo):
    import asyncio

    mo.stop(form.value is None, mo.md("**Submit the form to start generating.**"))
    with mo.status.spinner(title="(0/3) Generating Answers...") as _spinner:
        mo.output.append(
            mo.md(f"""**Neutral answer**: 
            {generate(form.value["prompt"], 0.0, form.value["token_count"])["content"]}"""),
        )
        _spinner.update("(1/3) Generating Answers...")
        mo.output.append(
            mo.md(f"""**Positive answer**: 
        {generate(form.value["prompt"], form.value["coeff"], form.value["token_count"])["content"]}"""),
        )
        _spinner.update("(2/3) Generating Answers...")
        mo.output.append(
            mo.md(f"""**Negative answer**: 
            {generate(form.value["prompt"], -form.value["coeff"], form.value["token_count"])["content"]}"""),
        )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
