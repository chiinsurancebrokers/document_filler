# 📋 FormFill AI

Streamlit app που χρησιμοποιεί Claude Vision για αυτόματη συμπλήρωση εντύπων PDF.

## Εγκατάσταση

```bash
# 1. Εγκατάσταση Python dependencies
pip install -r requirements.txt

# 2. Εγκατάσταση poppler (για PDF → εικόνα μετατροπή)
# Linux:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: κατεβάστε από https://github.com/oschwartz10612/poppler-windows

# 3. Εκκίνηση
streamlit run app.py
```

## Χρήση

1. Εισάγετε το **Anthropic API Key** στο sidebar
2. Ανεβάστε το **κενό έντυπο** (π.χ. αίτηση ασφάλισης)
3. Ανεβάστε τα **έγγραφα πηγής** (άδεια κυκλοφορίας, ταυτότητα, κλπ.)
4. Πατήστε **Fill Form Automatically**
5. Κατεβάστε το συμπληρωμένο PDF

## Pipeline

| Φάση | Τι κάνει |
|------|----------|
| Phase 1 | Claude σαρώνει το κενό έντυπο → εντοπίζει όλα τα πεδία |
| Phase 2 | Claude διαβάζει τα έγγραφα πηγής → εξάγει δεδομένα |
| Phase 3 | Claude αντιστοιχεί δεδομένα με πεδία → παράγει οδηγίες |
| Phase 4 | reportlab εφαρμόζει το κείμενο στο αρχικό PDF |

## Τεχνικές Λεπτομέρειες

- **AI Model**: Claude Sonnet / Opus (επιλογή χρήστη)
- **PDF Processing**: pypdf + reportlab
- **Γραμματοσειρά**: DejaVuSans (υποστηρίζει Ελληνικά)
- **Συντεταγμένες**: % ποσοστά σελίδας (0.0–1.0) για ανεξαρτησία ανάλυσης

## Απαιτήσεις Συστήματος

- Python 3.11+
- poppler-utils (για pdf2image)
- DejaVuSans font (προεγκατεστημένο σε Ubuntu/Debian)
- Anthropic API key
