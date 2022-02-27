# Project structure:

- Data can be found in: https://www.kaggle.com/c/champs-scalar-coupling/data
- multifc_*.ipynb contains the training of different models generating embeddings for all the different targets.
- resume_training_*.ipynb contains code for fine-tuning the model for each specific target class.
- generate_embeddings.ipynb contains code for generating the embeddings used for the tabular models.
- DL_tabular_*.ipynb contains code for the models making use of tabularized data.
- ray_opt_model.py contains the script used for optimizing the hyperparameters of a model using ray.tune
- performance_evaluation.ipynb contains code for evaluating the performance of all the different models.
