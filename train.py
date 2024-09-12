import ultralytics

ultralytics.checks()

# train the model
n_epochs = 30
bs = -1
verbose = True
rng = 0
validate = True
imgsz = 640

model = YOLO('yolov8s-seg.pt') # small version of model

results = model.train(
    data='fip/data.yaml',
    epochs=n_epochs,
    batch=bs,
    verbose=verbose,
    seed=rng,
    val=validate,
    imgsz=imgsz
)


