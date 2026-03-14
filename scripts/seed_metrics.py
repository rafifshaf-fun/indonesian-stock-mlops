import requests
import time

API_URL = "http://localhost:8000/predict"

TICKERS = [
    "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK",
    "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK",
    "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BREN.JK",
    "BRIS.JK", "BRPT.JK", "CPIN.JK", "CTRA.JK", "EXCL.JK",
    "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK",
    "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK",
    "MAPA.JK", "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK",
    "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK",
    "SMRA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
]

print("Fetching latest tickers data against training model and returning signal...")
for ticker in TICKERS:
    try:
        response = requests.post(
            "http://localhost:8000/predict", 
            json={"ticker": ticker}, 
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {ticker}: {data['signal']} ({data['probability_up']*100:.2f}%)")
        else:
            print(f"⚠️ {ticker}: FAILED - {response.text}")
            
    except requests.exceptions.ReadTimeout:
        print(f"⚠️ {ticker}: TIMEOUT - Yahoo Finance might be rate-limiting.")
    except Exception as e:
        print(f"⚠️ {ticker}: ERROR - {str(e)}")
        
    # THE CRITICAL FIX: Sleep for 2 seconds between requests
    # This stops Yahoo Finance from blocking your IP address!
    time.sleep(2)