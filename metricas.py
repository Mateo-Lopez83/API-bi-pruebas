from pipeline import RegressionPipeline

num_features = ['r', 'g', 'mjd', 'ra', 'dec']
cat_features = ['class']

pipeline = RegressionPipeline("Regresion_train_data.csv", num_features, cat_features)
pipeline.train()
pipeline.save_model("prueba_modelo.joblib")
print(pipeline.evaluate())
