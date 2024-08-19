# Excercise 1: Image Classification

## Develop
### Cài đặt

- Tải notebook GO_InternTest.ipynb và upload lên Google Colab hoặc Kaggle.
- Chọn môi trường GPU để có thể huấn luyện và sử dụng mô hình để phân loại ảnh nhanh nhất.

### Huấn luyện

- Chạy các đoạn code nằm trong các mục:

    - Import necessary libraries (import các thư viện cần thiết)
    - Load dataset (tải và chuẩn bị dataset cho việc huấn luyện mô hình)
    - Create model (tạo mô hình)
    - Train model (huấn luyện và lưu mô hình trong thư mục ./GO_intern/baseline)

- Chạy tiếp các đoạn code nằm trong mục `Getting last 1,2 percent` để có mô hình có kết quả cao hơn (được luu trong ./GO_intern/sgd_to_adam và ./GO_intern/sgd_to_rmsprop)

- Có thể vào file explorer của Google Colab hoặc Kaggle để tải các checkpoint.

### Inference

Sử dụng `keras.models.load_model('path/to/the/model/[model name].keras', compile=False, safe_mode=True)` để load mô hình lưu trong các thư mục trên. Sử dụng đoạn code dưới đây để xử lý 1 ảnh và đưa cho model phân loại :

    img = Image.open(requests.get('[link-to-image]', stream=True).raw) # Tải ảnh từ trên mạng.
    img = keras.utils.img_to_array(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, 0)
    print(f'Prediction: {"cat" if model.predict(img).argmax(axis=-1)[0] == 0 else "dog"}')

## Deploy

