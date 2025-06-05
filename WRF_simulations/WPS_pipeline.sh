#!/bin/bash
# This script runs the WPS program for a specific namelist the main script loop.
root_dir="$1"
WPS_source_dir="$2"
geogrid_dir="$3"
start_date="$4"
end_date="$5"
grib_source_dir="$6"
grib_prefix="$7"
Vtable="$8"

# instead of copying the WPS_source_dir, do rsync, so that only the changed files are copied.
rsync -av --exclude='namelist.wps' --exclude='Vtable' $WPS_source_dir/* .   # copy the WPS source files to the run directory
ln -sf $geogrid_dir/geo_em.d0* .   # link the geogrid files to the run directory
cp $geogrid_dir/namelist.wps .     # copy the namelist.wps file to the run directory
sed -i -e "s/ start_date =/ start_date =$start_date,$start_date,$start_date, /g" namelist.wps    # change the start_date in the namelist.wps file
sed -i -e "s/ end_date =/ end_date =$end_date,$end_date,$end_date, /g" namelist.wps    # change the end_date in the namelist.wps file
# add an if statement to check if grib_prefix is ERA5 or GFS, then 
# link the Vtable file to the run directory

# execute this block if the grib_prefix is CERRA
if [ $grib_prefix == 'CERRA' ]; then
    ln -sf "$root_dir/Vtable.CERRA" Vtable   # link the Vtable file to the run directory
    sed -i -e "s/ prefix =/ prefix ='CERRA'/g" namelist.wps # change the prefix in the namelist.wps file
    ./link_grib.csh $grib_source_dir'/CERRA_'* # link the grib files to the run directory
    ./ungrib.exe    # run the ungrib program

    sed -i -e "s/ prefix ='CERRA'/ prefix ='ERA5'/g" namelist.wps # change the prefix in the namelist.wps file
    ln -sf ungrib/Variable_Tables/Vtable.ERA-interim.pl Vtable # link the Vtable file to the run directory
    ./link_grib.csh $grib_source_dir'/ERA5_'*    # link the grib files to the run directory
    ./ungrib.exe    # run the ungrib program
    ./metgrid.exe   # run the metgrid program
else
    ln -sf "ungrib/Variable_Tables/"$Vtable Vtable   # link the Vtable file to the run directory
    ./link_grib.csh $grib_source_dir'/'*'.grb' # link the grib files to the run directory
    ./ungrib.exe    # run the ungrib program
    ./metgrid.exe   # run the metgrid program
fi

mkdir metfiles
mv met_em* metfiles

rm -r CERRA:*   # remove the CERRA files
rm -r ERA5:*    # remove the ERA5 files
rm -r GFS:*
rm -r GRIBFILE* # remove the GRIB files

