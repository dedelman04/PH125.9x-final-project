# PH125.9x-final-project
IDV "Choose your own" project for HX Capstone course

Project contents:
- report.Rmd: R markup file for creation of the report
- dataprep.R: code that mirrors report.Rmd to prep the data in a local instance; does not include any plots created in report.Rmd
- train_values.csv: file containing predictors
- train_labels.csv: file containing known outcomes
- results.csv: file containing RMSE and tuning parameters of the predictive models used in the report; this is used within report.Rmd to display the RMSE and/or tuning parameters so the models do not have to be trained and run each time report.Rmd is knitted (would take a couple of hours each time)
- Heart Disease Mortality.pdf: final report as a results of knitting report.Rmd to pdf
