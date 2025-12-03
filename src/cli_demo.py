"""Command-line demo for phishing email detection."""

from src.inference import predict_email_bert, predict_email_classic


def main() -> None:
    print("=== Phishing Email Detector ===")
    text = input("Paste the email text:\n> ")
    model_choice = input("Choose model [c]lassic / [b]ert (default classic): ").strip().lower() or "c"

    if model_choice.startswith("b"):
        label, proba, explanation = predict_email_bert(text)
        model_name = "BERT"
    else:
        label, proba, explanation = predict_email_classic(text)
        model_name = "Classic TF-IDF"

    print(f"\nModel: {model_name}")
    print(f"Predicted label: {'Phishing' if label == 1 else 'Ham/Safe'}")
    print(f"Probability phishing: {proba:.3f}")
    if explanation:
        print("\nTop explanation terms:")
        for term, val in explanation:
            print(f"  {term}: {val:+.3f}")
    else:
        print("\n(No explanation available)")


if __name__ == "__main__":
    main()
