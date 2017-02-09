I am interested in identifying important factors leading to hospital
early readmission for diabetic patients.According to a study by Medicare
Payment advisory committee, 17.6% of hospital admissions are readmitted
within 30 days of discharge, and more interestingly 76% of these
encounters are preventable [1,3]. Many hospitals were penalized a
minimum of 3% of their Medicare reimbursements for early readmissions in
2014[2,3]. Thus, hospitals need reliable patient-care strategies to
reduce early readmission rates and to improve patient health outcomes.
In a study of diabetes care management in 130 US hospitals, researchers
suggested the relationship between HbA1c measurements and reduced early
readmission rates in diabetic patients using multivariable logistic
regression.I am interested in applying a fuzzy-based model (a rule-based
approach that model human reasoning) to create a diagnosis protocol for
diabetic patients. The reference dataset is retrieved from UCI Machine
leaning Repository, the center for machine learning and intelligent
systems. The dataset contains 101,766 diabetic encounters and 55
features (demographics, diagnoses, diabetic medications, visit
information and payer code) from 130 US hospitals and integrated
delivery networks in 10 years (1999-2008). The preliminary dataset
contains missing values, redundant features and multiple encounters for
one patient (rows are not independent). Also, many categorical features
are transformed to the numerical one for machine learning analysis.
Later, I also need to perform Principal component analysis (PCA) to
reduce the number of available features based on the correlation
results. I used three training algorithms (K nearest neighbors, Decision
tree and logistic regression) to fit the relationship between features
and early readmission rate and evaluated the results using accuracy,
precision and F1-score metrics. The highest accuracy score is 91% with
precision score of 66%. [1] M.P.A. Committee, Report to Congress:
Promoting Greater Efficiency in Medicare, 2007.[2] C. for Medicare, M.
Services, Readmissions Reduction Program, August
2014.<http://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Readmissions-Reduction-Program.html>.[3]
J Futoma, J. Morris, J. Lucas, "A comparison of models for predicting
early hospital readmissions", Journal of Biomedical Informatics, vol.
56, pp. 229-238, 2015.
