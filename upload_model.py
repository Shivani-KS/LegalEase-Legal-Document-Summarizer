from huggingface_hub import upload_folder

upload_folder(
    repo_id="aneesh0312/LegalEase-Pegasus",  
    folder_path="legal-pegasus-final-69",
    repo_type="model"
)
