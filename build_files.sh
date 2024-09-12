echo "BUILD START"
python3.12 -m pip install -r requiremnets.txt
python3.12 manage.py collectstaic --noinput --clear
echo "BUILD END"