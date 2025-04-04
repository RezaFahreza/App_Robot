import cv2
import numpy as np

class SymbolProcessor:
    def __init__(self):
        # Ubah threshold untuk perbandingan simbol
        self.similarity_threshold = 1.0
    
    def extract_reference_symbols(self, reference_image):
        # Simpan gambar untuk debugging
        cv2.imwrite("debug_images/debug_reference_area.png", reference_image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Periksa polaritas dan samakan (latar belakang hitam, simbol putih)
        white_pixel_percentage = np.sum(binary) / (binary.shape[0] * binary.shape[1] * 255)
        if white_pixel_percentage > 0.5:
            binary = 255 - binary
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours (noise)
        contours = [c for c in contours if 50 < cv2.contourArea(c) < 5000]  # Batasi ukuran maksimum
        
        # TAMBAHKAN URUTAN EKSPLISIT BERDASARKAN POSISI X
        # Ini memastikan urutan dari kiri ke kanan (A, B, C, D, E)
        sorted_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            sorted_positions.append((x, contour))
        
        # Sort berdasarkan posisi X secara ketat
        sorted_positions.sort(key=lambda item: item[0])
        contours = [item[1] for item in sorted_positions]
        
        # Pastikan hanya 5 simbol referensi
        if len(contours) > 5:
            contours = contours[:5]  # Ambil 5 pertama
            
        # Extract each symbol
        symbols = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            symbol_img = binary[y:y+h, x:x+w]
            
            # Calculate features
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Log transform for better numerical stability
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            # Horizontal and vertical projections
            h_proj = np.sum(symbol_img, axis=1) / 255
            v_proj = np.sum(symbol_img, axis=0) / 255
            
            symbols.append({
                'image': symbol_img,
                'contour': contour,
                'position': (x, y, w, h),
                'features': {
                    'hu_moments': hu_moments,
                    'h_proj': h_proj,
                    'v_proj': v_proj,
                    'aspect_ratio': float(w) / h,
                    'area': cv2.contourArea(contour)
                }
            })
            
            # Save symbol for debugging
            cv2.imwrite(f"debug_images/debug_reference_symbol_{i}.png", symbol_img)
            print(f"Ref symbol {i} position: {(x, y, w, h)}")
        
        return symbols
    
    def extract_inner_symbol(self, box_img):
        # Convert to grayscale
        gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY) if len(box_img.shape) == 3 else box_img
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Periksa polaritas dan samakan (latar belakang hitam, simbol putih)
        white_pixel_percentage = np.sum(binary) / (binary.shape[0] * binary.shape[1] * 255)
        if white_pixel_percentage > 0.5:
            binary = 255 - binary
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return binary
        
        # Get largest contour (likely the symbol)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask and extract symbol
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply mask to get clean symbol
        symbol = cv2.bitwise_and(binary, mask)
        
        return symbol
    
    def extract_question_symbols(self, question_image):
        # Simpan gambar untuk debugging
        cv2.imwrite("debug_images/debug_question_area.png", question_image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(question_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Periksa polaritas dan samakan (latar belakang hitam, simbol putih)
        white_pixel_percentage = np.sum(binary) / (binary.shape[0] * binary.shape[1] * 255)
        if white_pixel_percentage > 0.5:
            binary = 255 - binary
        
        # Find contours of boxes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size and shape (boxes should be rectangular)
        box_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Typical boxes have aspect ratio close to 1 (square)
            if 0.5 < aspect_ratio < 1.5 and cv2.contourArea(contour) > 500:
                box_contours.append((x, y, w, h, contour))
        
        # Sort boxes by x position (left to right)
        box_contours = sorted(box_contours, key=lambda b: b[0])
        
        # Extract symbols from inside boxes
        symbols = []
        for i, box in enumerate(box_contours):
            x, y, w, h = box[:4]
            
            # Extract the area inside the box with padding
            padding = 5
            inner_x = x + padding
            inner_y = y + padding
            inner_w = w - 2 * padding
            inner_h = h - 2 * padding
            
            # Make sure we're within bounds
            if inner_w <= 0 or inner_h <= 0:
                continue
                
            # Get the region of interest
            inner_roi = binary[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
            
            # Extract clean symbol using new method
            clean_symbol = self.extract_inner_symbol(inner_roi)
            
            # Find contours of the clean symbol
            clean_contours, _ = cv2.findContours(clean_symbol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if clean_contours:
                # Get largest contour (the symbol)
                inner_contour = max(clean_contours, key=cv2.contourArea)
                
                # Calculate features
                moments = cv2.moments(inner_contour)
                hu_moments = cv2.HuMoments(moments).flatten()
                
                # Log transform
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
                
                # Horizontal and vertical projections
                h_proj = np.sum(clean_symbol, axis=1) / 255
                v_proj = np.sum(clean_symbol, axis=0) / 255
                
                # Add symbol to list
                symbols.append({
                    'image': clean_symbol,
                    'contour': inner_contour,
                    'position': (inner_x, inner_y, inner_w, inner_h),
                    'features': {
                        'hu_moments': hu_moments,
                        'h_proj': h_proj,
                        'v_proj': v_proj,
                        'aspect_ratio': float(inner_w) / inner_h,
                        'area': cv2.contourArea(inner_contour)
                    }
                })
                
                # Save for debugging
                cv2.imwrite(f"debug_images/debug_question_symbol_{i}.png", clean_symbol)
        
        return symbols
    
    # Tambahkan metode compare_symbols ini
    def compare_symbols(self, symbol1, symbol2):
        # Resize ke ukuran standar untuk perbandingan
        size = (64, 64)  # Ukuran yang lebih besar untuk detail lebih baik
        img1 = cv2.resize(symbol1['image'], size)
        img2 = cv2.resize(symbol2['image'], size)
        
        # Perbandingan berbasis pixel (lebih detail)
        pixel_diff = np.sum(np.abs(img1 - img2)) / (size[0]*size[1]*255)
        
        # Contour matching
        try:
            contour_diff = cv2.matchShapes(symbol1['contour'], symbol2['contour'], cv2.CONTOURS_MATCH_I2, 0.0)
        except:
            contour_diff = 1.0  # Default high value if matching fails
        
        # Pencocokan template
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        template_score = 1.0 - result.max()
        
        # Kombinasikan dengan bobot yang disesuaikan
        similarity = (0.5 * pixel_diff) + (0.3 * contour_diff) + (0.2 * template_score)
        
        return similarity    

    def find_missing_symbol(self, ref_symbols, question_symbols):
        if not ref_symbols or not question_symbols:
            return None
        
        # Match question symbols to reference symbols
        matched_indices = set()
        
        print(f"Reference symbols: {len(ref_symbols)}")
        print(f"Question symbols: {len(question_symbols)}")
        
        # Buat matriks loss untuk semua kombinasi simbol pertanyaan dan referensi
        loss_matrix = {}
        for i, q_sym in enumerate(question_symbols):
            for j, r_sym in enumerate(ref_symbols):
                loss = self.compare_symbols(q_sym, r_sym)
                if j not in loss_matrix:
                    loss_matrix[j] = []
                loss_matrix[j].append(loss)
        
        # Untuk setiap simbol pertanyaan, temukan kecocokan terbaik
        for i, q_sym in enumerate(question_symbols):
            best_match_idx = -1
            best_match_loss = float('inf')
            
            for j, r_sym in enumerate(ref_symbols):
                loss = self.compare_symbols(q_sym, r_sym)
                
                if loss < best_match_loss:
                    best_match_loss = loss
                    best_match_idx = j
            
            print(f"Question symbol {i} matched with reference {best_match_idx} (loss: {best_match_loss:.4f})")
            
            # Simbol hanya dianggap cocok jika loss-nya di bawah threshold
            if best_match_loss < self.similarity_threshold:
                matched_indices.add(best_match_idx)
        
        # Find the missing index (symbol not in question)
        all_indices = set(range(len(ref_symbols)))
        missing_indices = all_indices - matched_indices
        
        print(f"Matched indices: {matched_indices}")
        print(f"Missing indices: {missing_indices}")
        
        if missing_indices:
            # Jika ada lebih dari satu indeks yang hilang
            if len(missing_indices) > 1:
                print(f"Multiple missing indices detected: {missing_indices}")
                
                # Kita perlu menganalisis semua loss untuk tiap indeks yang hilang
                min_loss_indices = {}
                for idx in missing_indices:
                    if idx in loss_matrix and loss_matrix[idx]:
                        # Ambil loss terkecil (kecocokan terbaik) untuk indeks ini
                        min_loss = min(loss_matrix[idx])
                        min_loss_indices[idx] = min_loss
                        print(f"Index {idx} min loss: {min_loss:.4f}")

                # Pilih indeks dengan min loss terkecil (kecocokan terbaik)
                if min_loss_indices:
                    # Indeks dengan kecocokan terbaik dianggap sebenarnya cocok
                    best_match_idx = min(min_loss_indices.items(), key=lambda x: x[1])[0]
                    
                    # Indeks yang TIDAK terpilih menjadi jawaban
                    missing_indices.remove(best_match_idx)
                    missing_idx = list(missing_indices)[0]  # Mengambil indeks yang tersisa
                    
                    print(f"Index {best_match_idx} has best match with min loss: {min_loss_indices[best_match_idx]:.4f}")
                    print(f"Selected missing index {missing_idx} as true missing symbol")
                else:
                    # Fallback: ambil indeks pertama jika tidak ada data loss
                    missing_idx = list(missing_indices)[0]
            else:
                missing_idx = list(missing_indices)[0]
            
            # Convert to A, B, C, D, E
            return chr(65 + missing_idx)
        
        return None