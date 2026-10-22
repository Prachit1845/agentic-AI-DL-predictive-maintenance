import os
import win32com.client

def convert_docx_to_pdf(docx_path, pdf_path):
    print(f"Converting {docx_path} to {pdf_path}...")
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path))
        # 17 is wdFormatPDF in Word COM
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
        doc.Close()
        print(f"Successfully created PDF: {pdf_path}")
    except Exception as e:
        print(f"Error during PDF conversion: {e}")
    finally:
        word.Quit()

if __name__ == "__main__":
    docx_file = "Agentic_Predictive_Maintenance_LSTM_Report.docx"
    pdf_file = "Agentic_Predictive_Maintenance_LSTM_Report.pdf"
    convert_docx_to_pdf(docx_file, pdf_file)
