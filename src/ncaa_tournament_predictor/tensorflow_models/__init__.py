# Splitting out a separate Tensorflow package; the intention was originally to
# use AutoML in Databricks so I didn't need to write my own model. However, due to limitations/issues
# in our Databricks sandbox environment, I can't use AutoML. I'm isolating this package
# to make a clean line between Databricks integration and roll-your-own-model-choose-your-own-adventure
# logic. Someday, I can compare this to the AutoML-generated model when environment issues are worked out
