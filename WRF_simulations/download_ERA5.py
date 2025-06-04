import sys
import cdsapi
import yaml

def download_era5_data(c, year: str, month: str, days: list):

    # Pressure-level variables: z, t, r
    c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type":   "reanalysis",
        "area":           "60.00/-20.00/40.00/20.00",
        "variable":       ["z", "t", "r"],
        "pressure_level": ["1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100",
                           "125", "150", "175", "200", "225", "250", "300", "350", "400",
                           "450", "500", "550", "600", "650", "700", "750", "775", "800",
                           "825", "850", "875", "900", "925", "950", "975", "1000"],
        "year":           year,
        "month":          month,
        "day":            days,
        "time":           [f"{h:02d}" for h in range(24)],
    }, f"PRES_SC_{year}_{month}_{days[0]}-{days[-1]}.grb")

    # Pressure-level variables: u, v
    c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type":   "reanalysis",
        "area":           "60.00/-20.00/40.00/20.00",
        "variable":       ["u", "v"],
        "pressure_level": ["1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100",
                           "125", "150", "175", "200", "225", "250", "300", "350", "400",
                           "450", "500", "550", "600", "650", "700", "750", "775", "800",
                           "825", "850", "875", "900", "925", "950", "975", "1000"],
        "year":           year,
        "month":          month,
        "day":            days,
        "time":           [f"{h:02d}" for h in range(24)],
    }, f"PRES_UVW_{year}_{month}_{days[0]}-{days[-1]}.grb")

    # Single-level variables
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type":   "reanalysis",
        "area":           "60.00/-20.00/40.00/20.00",
        "variable":       ["10u", "10v", "2t", "2d", "msl", "sp", "sst", "skt",
                           "stl1", "stl2", "stl3", "stl4", "slt", "swvl1", "swvl2", "swvl3", "swvl4",
                           "sd", "rsn", "lsm", "ci"],
        "year":           year,
        "month":          month,
        "day":            days,
        "time":           [f"{h:02d}" for h in range(24)],
    }, f"SFC_{year}_{month}_{days[0]}-{days[-1]}.grb")


if __name__ == "__main__":
	if len(sys.argv) < 5:
		print("Usage: python download_era5.py <cdsapi_file> <year> <month> <day1,day2,...>")
		sys.exit(1)

	cdsapirc_file = sys.argv[1]
	year = sys.argv[2]
	month = sys.argv[3]
	days = sys.argv[4].split(',')

	with open(cdsapirc_file, 'r') as f:
		credentials = yaml.safe_load(f)
	c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

	download_era5_data(c, year, month, days)