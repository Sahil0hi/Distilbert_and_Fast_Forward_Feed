

## How To Use
### The repository contains five main files:
- `model.py` is where you define your model
- `prep_data.py` is used to process the data for training
- `train.py` contains the code for training your model
- `test.py` contains a few useful functions to evaluate your solution
as well as the cross-validation testing we will use to pick winners.
- `main.py` outputs information about the data, your model architecture,
and your model performance. **Run this file and include a screenshot of the output
in your DevPost submission.**

### Basic Workflow
1. Make any desired alterations to the data, and update `prep_data.get_prepared_data()`
to return your version of the data if needed.
2. Create your model (change the `MyModel` class in `model.py`).
3. Test your model (using your own code, or one of the provided functions in `train.py`/`test.py`).
4. Repeat steps 1-3 until you have achieved the highest accuracy possible across the entire dataset.
5. Run `main.py` and include its output in your submission.


- Your model must take the form of a class in `model.py` with the name `MyModel` that extends `torch.nn.Module`.
- Do not include any data or model weights in your repository on GitHub.
- Do not modify `main.py` in any way.
- We should be able to reproduce your stated results exactly by
downloading the datasets you cited in your DevPost and placing them in the existing `data` folder, then running 
`main.py`. Make sure you haven't manually renamed any dataset files.
- You may use any open source model **as a component of your solution** (not your entire solution). Your code must download existing models itself,
without requiring API keys or authentication of any kind.
- Your model must be reasonably efficient. If it takes more than 10 minutes to train your model on a single A100 GPU,
or the model requires more than 4GB of GPU memory, you will be disqualified.
- You may not commit to your repository after 11:59pm on Monday, February 10th, 2025.
