from herbie import Herbie
import sys

# Get arguments from the command line
variable = sys.argv[1]
level = sys.argv[2]
year = sys.argv[3]
month = sys.argv[4]
day = sys.argv[5]
hour = sys.argv[6]
download_path = sys.argv[7]

H = Herbie(
    f"{year}-{month}-{day}T{hour}:00:00",
    model="rtma",
    product="anl",
    save_dir = download_path,
)

H.download(f"{variable}:{level}",verbose=False,overwrite=True)
