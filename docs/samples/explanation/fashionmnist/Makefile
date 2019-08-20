
model.h5 input_shape.joblib:
	python train.py

.PHONY: train
train: model.h5 input_shape.joblib

.PHONY: clean
clean:
	rm -f *.joblib
	rm -f *.h5

