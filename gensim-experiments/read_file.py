import PyPDF2

pdfFileObj = open('pdf-sample.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)


pageObj = pdfReader.getPage(0)
print(pageObj.extractText())