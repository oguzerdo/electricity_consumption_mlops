
prep:
	python scripts/preprocess_data.py

tuning:
	python scripts/tuning.py

register:
	python scripts/register_model.py

register-best:
	python scripts/register_model.py --top_n 1

start:
	uvicorn main:app --reload