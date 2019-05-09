from dbrnna import run_dbrnna_chronological_cv, transfer_learning
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

gc2mc_accuracy = transfer_learning(
    train_dataset="google_chromium",
    test_dataset="mozilla_core",
    min_train_samples_per_class=20,
)
print("gc2mc_accuracy:", gc2mc_accuracy)


mc2mf_accuracy = transfer_learning(
    train_dataset="mozilla_core",
    test_dataset="mozilla_firefox",
    min_train_samples_per_class=20,
)
print("mc2mf_accuracy:", mc2mf_accuracy)
