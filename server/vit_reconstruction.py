import torch
import torch.nn as nn

# ==============================================================================
# TẦNG 1: XỬ LÝ DỮ LIỆU & BIẾN ĐỔI ẢNH (DATA PIPELINE)
# ==============================================================================
# Trạng thái trong Workflow: PROCESSING -> Extract Patches từ frame đã crop.
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        # Linear Projection: Phẳng hóa các mảnh ảnh và chuyển thành vector nhúng
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        
        # Position Embedding & Class Token
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))

    def forward(self, x):
        b, c, h, w = x.shape
        # Chia ảnh thành patch và flatten: [B, C, H, W] -> [B, Emb_size, Num_patches] -> [B, Num_patches, Emb_size]
        x = self.projection(x).flatten(2).transpose(1, 2)
        
        # Chèn Class Token vào đầu
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Thêm Position Embedding
        x += self.positions
        return x

# ==============================================================================
# TẦNG 2: KIẾN TRÚC MẠNG & CƠ CHẾ CHÚ Ý (CORE ViT ARCHITECTURE)
# ==============================================================================
# Trạng thái trong Workflow: PROCESSING -> Transformer học liên kết không gian 3D.
class TransformerBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        # Multi-Head Self-Attention (MHSA)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        
        # Mạng Feed-Forward (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * forward_expansion, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Kết nối tắt (Residual Connection) + Layer Normalization
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x += res
        
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += res
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, num_classes=1000):
        super().__init__()
        # Khởi tạo Tầng 1
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        
        # Khởi tạo Tầng 2 (Các khối Transformer Encoder)
        self.transformer = nn.Sequential(*[TransformerBlock(emb_size) for _ in range(depth)])
        
        # MLP Head (Dự đoán đặc trưng 3D hoặc phân loại)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes) 
            # Note: Với bài toán 3D Recon, đầu ra ở đây có thể đưa vào 3D Decoder thay vì classification.
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        # Chỉ lấy dữ liệu từ [CLS] token để đưa vào dự đoán
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

# ==============================================================================
# TẦNG 3: HUẤN LUYỆN & TỐI ƯU HÓA (TRAINING & OPTIMIZATION)
# ==============================================================================
# Hàm này dùng để gọi trong luồng huấn luyện của backend
def setup_training_pipeline(model):
    import torch.optim as optim
    
    # Optimizer: Dùng AdamW như khuyến nghị cho Transformer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Loss Function: Tùy bài toán (Ví dụ nếu dùng ViT để trích xuất feature 3D -> Dùng Chamfer Distance Loss)
    # Ở đây mô phỏng hàm loss cơ bản
    criterion = nn.MSELoss() # Dùng MSE để so khớp vector đặc trưng 3D dự đoán với Ground Truth
    
    return optimizer, criterion

# Hàm mô phỏng pipeline inference gọi từ server FastAPI (Workflow: UPLOADING -> PROCESSING)
def process_frame_for_reconstruction(image_tensor):
    """
    Giả lập: Hàm này nhận image tensor từ bước YOLO (Đã crop vật thể)
    và đưa vào ViT để tạo vector nhúng 3D.
    """
    model = VisionTransformer(img_size=224, patch_size=16, emb_size=768, depth=6, num_classes=512) # 512 là feature dim cho 3D
    model.eval()
    
    with torch.no_grad():
        features_3d = model(image_tensor)
        
    return features_3d
