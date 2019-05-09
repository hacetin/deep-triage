from dbrnna import run_dbrnna_chronological_cv
from preprocess import preprocess_all_datasets


preprocess_all_datasets()

gc_result_dict = run_dbrnna_chronological_cv(
    dataset_name="google_chromium", min_train_samples_per_class=20, num_cv=10
)
print("gc_result_dict:", gc_result_dict)

mc_result_dict = run_dbrnna_chronological_cv(
    dataset_name="mozilla_core", min_train_samples_per_class=20, num_cv=10
)
print("mc_result_dict:", mc_result_dict)

mf_result_dict = run_dbrnna_chronological_cv(
    dataset_name="mozilla_firefox", min_train_samples_per_class=20, num_cv=10
)
print("mf_result_dict:", mf_result_dict)
