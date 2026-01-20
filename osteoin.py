import sys
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
import platform
import subprocess
import tempfile
from PIL import Image, ImageWin, ImageEnhance

# 운영체제 확인
CURRENT_OS = platform.system()

# 윈도우일 때만 윈도우 인쇄 기능 불러오기
if CURRENT_OS == 'Windows':
    import win32print
    import win32ui

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class DocumentScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"스캔 마스터 ({CURRENT_OS} 버전)")
        self.setGeometry(100, 100, 500, 400)
        self.setAcceptDrops(True) 

        self.label = QLabel("파일을 드래그하세요.\n(PDF, JPG, PNG)\n\n[고화질 & 하단 정렬]", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; color: #333; border: 3px dashed #0078d7; background-color: #f5faff;")
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext == '.pdf':
                self.process_pdf(f)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                self.process_single_file(f)
            else:
                self.label.setText(f"지원하지 않는 파일: {ext}")

    def process_pdf(self, path):
        try:
            doc = fitz.open(path)
            total = len(doc)
            for i, page in enumerate(doc):
                self.label.setText(f"PDF 고화질 변환 중... ({i+1}/{total})")
                QApplication.processEvents()
                
                # 고화질 렌더링 (약 300 DPI)
                zoom = 4.17 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_data = img_data.reshape(pix.h, pix.w, pix.n)
                
                if pix.n >= 4: img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                else: img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

                processed = self.process_cv_image(img_data)
                if processed: self.print_image(processed)
            self.label.setText(f"완료 ({total}장)")
        except Exception as e:
            self.label.setText(f"오류: {e}")

    def process_single_file(self, path):
        self.label.setText("고화질 처리 중...")
        QApplication.processEvents()
        try:
            # 맥/윈도우 호환 파일 읽기
            img_array = np.fromfile(path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            
            if image is not None:
                processed = self.process_cv_image(image)
                if processed:
                    self.print_image(processed)
                    self.label.setText("완료! 다음 파일 대기 중")
            else:
                self.label.setText("이미지 로드 실패")
        except Exception as e:
             self.label.setText(f"오류: {e}")

    def process_cv_image(self, image):
        try:
            orig = image.copy()
            h_orig, w_orig = image.shape[:2]
            
            # 윤곽선 찾기 (다운샘플링)
            ratio = 1500 / max(h_orig, w_orig)
            if ratio < 1:
                small_img = cv2.resize(image, None, fx=ratio, fy=ratio)
            else:
                small_img = image.copy()
                ratio = 1

            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blur, 75, 200)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

            screenCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if cv2.contourArea(c) / (small_img.shape[0] * small_img.shape[1]) > 0.2:
                        screenCnt = approx
                        break
            
            # 투시 변환
            if screenCnt is not None:
                screenCnt = screenCnt.astype("float32") / ratio
                warped = self.four_point_transform(orig, screenCnt.reshape(4, 2))
            else:
                warped = orig

            # 강제 세로 모드
            h, w = warped.shape[:2]
            if w > h: warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            # 고화질 보정 (Gamma + Contrast)
            if len(warped.shape) == 3:
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            dilated = cv2.dilate(warped, np.ones((7,7), np.uint8))
            bg_blur = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(warped, bg_blur)
            norm_img = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            gamma = 1.5
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            enhanced = cv2.LUT(norm_img, lookUpTable)

            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            final = cv2.filter2D(enhanced, -1, sharpen_kernel)
            
            return Image.fromarray(final)

        except Exception as e:
            print(f"이미지 처리 에러: {e}")
            return None

    def four_point_transform(self, image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
        widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
        heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([[0, 0],[maxWidth-1, 0],[maxWidth-1, maxHeight-1],[0, maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

    def print_image(self, pil_image):
        if pil_image.width > pil_image.height:
            pil_image = pil_image.rotate(90, expand=True)

        if CURRENT_OS == 'Windows':
            self.print_windows(pil_image)
        elif CURRENT_OS == 'Darwin': # Mac
            self.print_mac(pil_image)
        else:
            print("지원하지 않는 OS")

    def print_windows(self, pil_image):
        try:
            printer = win32print.GetDefaultPrinter()
            hDC = win32ui.CreateDC()
            hDC.CreatePrinterDC(printer)
            HORZRES, VERTRES = 8, 10
            page_w = hDC.GetDeviceCaps(HORZRES)
            page_h = hDC.GetDeviceCaps(VERTRES)
            
            img_w, img_h = pil_image.size
            ratio = min(page_w/img_w, page_h/img_h)
            new_w, new_h = int(img_w*ratio), int(img_h*ratio)
            
            x_off = (page_w - new_w) // 2
            y_off = page_h - new_h # 하단 정렬

            hDC.StartDoc("High Quality Scan")
            hDC.StartPage()
            dib = ImageWin.Dib(pil_image)
            dib.draw(hDC.GetHandleOutput(), (x_off, y_off, x_off+new_w, y_off+new_h))
            hDC.EndPage()
            hDC.EndDoc()
            hDC.DeleteDC()
        except Exception as e:
            print(f"Win Print Error: {e}")

    def print_mac(self, pil_image):
        try:
            # 맥은 lpr 명령어로 인쇄. 하단 정렬을 위해 캔버스 생성
            # A4 @ 300 DPI
            a4_w, a4_h = 2480, 3508 
            canvas = Image.new('L', (a4_w, a4_h), 255)

            img_w, img_h = pil_image.size
            ratio = min(a4_w / img_w, a4_h / img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            
            resized_img = pil_image.resize((new_w, new_h), Image.LANCZOS)
            
            x_off = (a4_w - new_w) // 2
            y_off = a4_h - new_h # 하단 정렬
            
            canvas.paste(resized_img, (x_off, y_off))
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                canvas.save(temp.name, format="PNG")
                subprocess.run(['lpr', '-o', 'fit-to-page', temp.name])
                print("Mac 인쇄 전송 완료")
        except Exception as e:
            print(f"Mac Print Error: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DocumentScannerApp()
    window.show()
    sys.exit(app.exec_())