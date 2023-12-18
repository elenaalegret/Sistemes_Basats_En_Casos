# Project 2 - Book Recommendation System

## Authors
- Elena Alegret Regalado
- Júlia Orteu Aubach
- Isabel Garcia De La Fuente
- Àlex Miquel Casanovas Cordero

## Overview
This repository contains the second project for the SBC course. The project focuses on the implementation of a Case-Based Reasoning (CBR) system for book recommendations.

## Project Structure

- **`REPORT.pdf`:** Project report in PDF format. It provides detailed documentation following the methodology phases: identification, conceptualization, formalization, implementation, and testing.
- **`Enunciat.pdf`**: Project statement.

- **`/Code`**: This directory houses the Python implementation code of the CBR system.
	- `Functions.py`: Python script containing custom functions used in the implementation.
	- `cbr.py`: Python script containing the CBR system implementation with 4 R.
	- `recomanador_form.py`: Contains the code of visualization of the interface and the user interaction.
	- `/Notebooks_Implementation_and_Experimentation`: Directory containing Jupyter notebooks with the implementation and experimentation of the system.
	- `Decision_Tree.pkl`: Pickle file containing the trained Decision Tree model.
	- `decision_tree`: Decision Tree model file.

- **`/Ontologia`**: This directory houses the Ontology Preprocessing.
	- `/Preprocessing_Notebooks`: Directory containing Jupyter notebooks for ontology preprocessing.
	- `/Other_Dataframes`: Directory containing additional dataframes related to the ontology and the original ontology CLIPS file.
	- `/Experimentation_Dataframes`: Directory containing the extended dataframes for the experimentation.
	- `/Initial_Dataframes`: Directory containing initial versions of dataframes for the initialization of the system.
	- `Books.csv`, `Cases.csv`, `Users.csv`: CSV files containing important ontology-related data.

- **`/Evaluació`**: Directory that stores Evaluation templates.
	- `/Plantilles`: Directory containing templates for evaluation in Catalan, Spanish, and English languages.
	- `/Evaluacions`: Directory containing real user evaluations.

## Additional Information
---
- For detailed documentation, please refer to the `REPORT.pdf`.
- To run the system, it is mandatory to have the `Streamlite` library installed:
	- Download command: `$ pip install streamlit`
	- Execution of the code:
		1. Go into the `/Code` directory
		2. Command: `$ streamlit run recomanador_form.py`

