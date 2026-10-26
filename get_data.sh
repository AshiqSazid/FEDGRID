#!/usr/bin/env bash
# Downloads the public FedGrid-RL datasets into data/raw. Safe to re-run: files already present are skipped.
set -euo pipefail
cd "$(dirname "$0")"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
R=data/raw

get() { # url dest
  if [ -s "$2" ]; then echo "have   $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "get    $2"
  if curl -fsSL -A "$UA" --retry 3 -o "$2.part" "$1"; then mv "$2.part" "$2"
  else rm -f "$2.part"; echo "FAILED $1  (open it in a browser and save to $2)"; fi
}

# 1. CSIRO representative LV feeders (OpenDSS format)
get https://github.com/csiro-energy-systems/RepresentativeLVNetworks/archive/refs/heads/master.zip \
    $R/csiro/RepresentativeLVNetworks.zip

# 2. Ausgrid Solar Home Electricity Data, July 2010 to June 2013.
# Ausgrid's own page and data.nsw.gov.au return 404 (Oct 2026), so this uses Pierre Haessig's mirror of the three yearly files.
get https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip $R/ausgrid/Ausgrid_solar_home_data.zip

# 3. AEMO 5 minute price and demand, Queensland, 2025
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  get "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_2025${m}_QLD1.csv" \
      $R/aemo/PRICE_AND_DEMAND_2025${m}_QLD1.csv
done

# 4. Network tariff guides (Toowoomba is Ergon, Brisbane is Energex)
get https://www.energex.com.au/__data/assets/pdf_file/0009/1651554/Ergon-Energy-Network-Tariff-Guide-2025-26.pdf \
    $R/tariffs/Ergon-Energy-Network-Tariff-Guide-2025-26.pdf
get https://www.energex.com.au/__data/assets/pdf_file/0006/1651551/Energex-Network-Tariff-Guide-2025-26.pdf \
    $R/tariffs/Energex-Network-Tariff-Guide-2025-26.pdf

# 5. NASA POWER hourly irradiance and temperature, 2025.
# Toowoomba is the target. Sydney is where the Ausgrid homes are, so the pair gives the solar rescaling ratio.
nasa() { echo "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude=$1&latitude=$2&start=20250101&end=20251231&format=CSV"; }
get "$(nasa 151.95 -27.56)" $R/weather/nasa_power_toowoomba_2025.csv
get "$(nasa 151.21 -33.87)" $R/weather/nasa_power_sydney_2025.csv
# Long term monthly mean irradiance. src/prep/weather.py uses these for the Sydney to Toowoomba ratio, so one odd year doesn't skew it.
clim() { echo "https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude=$1&latitude=$2&format=CSV"; }
get "$(clim 151.95 -27.56)" $R/weather/nasa_power_climatology_toowoomba.csv
get "$(clim 151.21 -33.87)" $R/weather/nasa_power_climatology_sydney.csv

# 7. AEMO minimum system load thresholds and QED Q3 2025 (reference documents for the MSL scenarios)
get https://www.aemo.com.au/-/media/files/initiatives/der/managing-minimum-system-load/2025-spring-and-summer-minimum-system-load-thresholds-fact-sheet.pdf \
    $R/aemo/2025-spring-and-summer-minimum-system-load-thresholds-fact-sheet.pdf
get https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q3-2025.pdf \
    $R/aemo/qed-q3-2025.pdf

# ponytail: item 6 (CityLearn datasets) downloads itself on `pip install citylearn==3.0.2` first use.

for z in $R/csiro/*.zip $R/ausgrid/*.zip; do
  [ -s "$z" ] && unzip -n -q "$z" -d "$(dirname "$z")"
done
echo done
