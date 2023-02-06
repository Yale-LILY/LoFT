echo "==> Processing SASP Results"
python run_process_sasp_results.py
echo "==> Translating Programs (defined by SASP) to Logic Forms"
python run_transform_programs_to_lf.py
echo "==> Validating Processed Results"
python run_validate_sasp_results.py