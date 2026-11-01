"""Real PaddleOCR smoke test using synthetic data; no patient report is uploaded."""
import io
from PIL import Image, ImageDraw, ImageFont
from report_ocr import extract_report

def synthetic_report():
    image = Image.new('RGB', (1400, 650), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 32)
    for text, y in [('SYNTHETIC LAB REPORT - NOT A PATIENT', 40),
                    ('Test                         Result       Unit           Reference', 140),
                    ('Glucose                   98.1          mg/dL        70 - 100', 230),
                    ('CRP                         <5.0          mg/L          <5.0', 320),
                    ('Hemoglobin             13.2          g/dL           12.0 - 16.0', 410)]:
        draw.text((50, y), text, font=font, fill='black')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return image_bytes.getvalue()


def main():
    image_bytes = io.BytesIO(synthetic_report())
    result = extract_report(image_bytes.getvalue(), 'synthetic.png')
    print(result['text'])
    assert result['requires_review']
    for value in ('98.1', '13.2', '<5.0'):
        assert value in result['text'], f'Missing exact value: {value}'
    print('PASS: exact decimals and inequality retained; review required.')


if __name__ == '__main__':
    main()
