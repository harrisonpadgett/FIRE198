import requests
import csv
import datetime
import time

BASE_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
DETAIL_URL_TEMPLATE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{}.geojson"
LOSS_JSON_TEMPLATE = "https://earthquake.usgs.gov/product/losspager/{}/{}/{}/json/losses.json"
ALERT_JSON_TEMPLATE = "https://earthquake.usgs.gov/product/losspager/{}/{}/{}/json/alerts.json"

limit = 200
offset = 923
page = 0
total_written = 0
total = 0

starttime = (datetime.datetime.utcnow() - datetime.timedelta(days=3650)).strftime("%Y-%m-%d")
endtime = datetime.datetime.utcnow().strftime("%Y-%m-%d")

with open("us_earthquakes_with_losses_new.csv", "w", newline="") as csvfile:
    fieldnames = [
        "id", "time", "place", "mag", "alert",
        "fatalities", "economic_loss_usd",
        "fatality_alert", "economic_alert",
        "fatality_green", "fatality_yellow", "fatality_orange", "fatality_red",
        "economic_green", "economic_yellow", "economic_orange", "economic_red"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    while True:
        params = {
            "format": "geojson",
            "starttime": starttime,
            "endtime": endtime,
            "minmagnitude": 4.0,
            "orderby": "time",
            "limit": limit,
            "offset": offset,
            "minlatitude": 24.396308,
            "maxlatitude": 49.384358,
            "minlongitude": -125.0,
            "maxlongitude": -66.93457
        }

        print(f"\n🔄 Fetching page {page+1} with offset={offset}...")
        response = requests.get(BASE_QUERY_URL, params=params)
        response.raise_for_status()
        events = response.json().get("features", [])

        if response.status_code == 429:
            print("Rate limited. Waiting 2 mins.")
            time.sleep(120)

        if not events:
            print("✅ No more events found. Stopping.")
            break

        for event in events:
            event_id = event["id"]
            total += 1

            try:
                detail_url = DETAIL_URL_TEMPLATE.format(event_id)
                detail_resp = requests.get(detail_url)
                detail_resp.raise_for_status()
                detail_data = detail_resp.json()
            except Exception as e:
                print(f"⚠️ Failed to fetch full event data for {event_id}: {e}")
                continue

            props = detail_data.get("properties", {})
            products = props.get("products", {})

            fatality_count = -1
            economic_loss = -1
            fatality_alert = None
            economic_alert = None
            fatality_probs = {"green": -1, "yellow": -1, "orange": -1, "red": -1}
            economic_probs = {"green": -1, "yellow": -1, "orange": -1, "red": -1}

            for product_key in ["losspager", "pager"]:
                if product_key in products:
                    latest = max(products[product_key], key=lambda v: v.get("updateTime", 0))
                    product_code = latest.get("code", event_id)
                    author = latest.get("source", "").lower()
                    update_time = str(latest.get("updateTime", "")).strip()

                    if not (author and update_time):
                        continue

                    # Try losses.json
                    try:
                        loss_url = LOSS_JSON_TEMPLATE.format(product_code, author, update_time)
                        loss_resp = requests.get(loss_url)
                        if loss_resp.status_code == 200 and "application/json" in loss_resp.headers.get("Content-Type", ""):
                            loss_data = loss_resp.json()
                            fatality_count = loss_data.get("empirical_fatality", {}).get("total_fatalities", -1)
                            economic_loss = loss_data.get("empirical_economic", {}).get("total_dollars", -1)
                    except Exception as e:
                        print(f"⚠️ Error fetching losses.json for {event_id}: {e}")

                    # Try alerts.json (from binned probabilities)
                    try:
                        alert_url = ALERT_JSON_TEMPLATE.format(product_code, author, update_time)
                        alert_resp = requests.get(alert_url)
                        if alert_resp.status_code == 200 and "application/json" in alert_resp.headers.get("Content-Type", ""):
                            alert_data = alert_resp.json()
                            fatality_alert = alert_data.get("fatality", {}).get("level")
                            economic_alert = alert_data.get("economic", {}).get("level")

                            # Sum probabilities per color
                            for bin in alert_data.get("fatality", {}).get("bins", []):
                                color = bin.get("color")
                                if color in fatality_probs and fatality_probs[color] == -1:
                                    fatality_probs[color] = 0
                                if color in fatality_probs:
                                    fatality_probs[color] += float(bin.get("probability", 0))

                            for bin in alert_data.get("economic", {}).get("bins", []):
                                color = bin.get("color")
                                if color in economic_probs and economic_probs[color] == -1:
                                    economic_probs[color] = 0
                                if color in economic_probs:
                                    economic_probs[color] += float(bin.get("probability", 0))
                        elif alert_resp.status_code != 200:
                            print(f"❌ alerts.json missing or bad for {event_id}")

                    except Exception as e:
                        print(f"⚠️ Error fetching alerts.json for {event_id}: {e}")

                    break  # found losspager/pager with valid structure

            if fatality_count != -1 or economic_loss != -1:
                writer.writerow({
                    "id": event_id,
                    "time": props.get("time"),
                    "place": props.get("place"),
                    "mag": props.get("mag"),
                    "alert": props.get("alert"),
                    "fatalities": fatality_count,
                    "economic_loss_usd": economic_loss,
                    "fatality_alert": fatality_alert,
                    "economic_alert": economic_alert,
                    "fatality_green": fatality_probs.get("green", -1),
                    "fatality_yellow": fatality_probs.get("yellow", -1),
                    "fatality_orange": fatality_probs.get("orange", -1),
                    "fatality_red": fatality_probs.get("red", -1),
                    "economic_green": economic_probs.get("green", -1),
                    "economic_yellow": economic_probs.get("yellow", -1),
                    "economic_orange": economic_probs.get("orange", -1),
                    "economic_red": economic_probs.get("red", -1)
                })
                csvfile.flush()
                total_written += 1
                print(f"[{total}] ✅ Wrote {event_id} with data. Total written: {total_written}")
            else:
                print(f"[{total}] [SKIP] No loss data found for {event_id}.")

        offset += limit
        page += 1
        time.sleep(1)

print(f"\n✅ Done! Wrote {total_written} earthquakes to CSV.")