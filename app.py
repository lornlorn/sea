from flask import Flask, render_template, request, jsonify
import os
import glob
import cv2
import base64
import openai  # 修改导入方式
import shutil
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import datetime  # 添加datetime导入
import random
from flask_cors import CORS  # 添加 CORS 支持
import gc  # 添加垃圾回收模块

# 加载环境变量
load_dotenv()

# 获取API密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")

# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set")
# if not qwen_api_key:
#     raise ValueError("QWEN_API_KEY environment variable is not set")

# 配置 OpenAI
openai.api_key = qwen_api_key  # 设置默认 API key

# 修改 Flask 应用初始化
app = Flask(__name__)

# 添加 CORS 支持
CORS(app)

# 配置应用
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max-limit
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# 配置 OpenCV 以减少内存使用
cv2.setNumThreads(1)  # 限制 OpenCV 线程数

# 默认提示词
DEFAULT_PROMPT = "这些是我想上传的视频帧。它描述了中国海域的海浪高度。请生成海洋预报的解说词，以便我可以随视频一起上传。解说词不需要包含具体日期，颜色和海浪高度数字。解说词必须是中文的，大约150字左右。"

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 添加请求后清理函数
@app.after_request
def after_request(response):
    gc.collect()  # 强制垃圾回收
    return response

def process_with_openai(prompt, base64Frames):
    try:
        # 使用旧版本的客户端初始化方式
        openai.api_key = openai_api_key  # 确保使用正确的 API key
        openai.api_base = "https://api.openai.com/v1"  # 设置 OpenAI API base URL
        
        # 限制图片数量
        max_images = 5  # 限制最大图片数量
        selected_frames = base64Frames[:max_images]
        print(f"Using {len(selected_frames)} images out of {len(base64Frames)} total frames")
        
        # 使用旧版本的 API 调用格式
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame}"
                                }
                            }
                            for frame in selected_frames
                        ]
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        raise

def process_with_qwen(prompt, weather, base64Frames):
    print("Starting Qwen API call...")
    try:
        # 使用旧版本的客户端初始化方式
        openai.api_key = qwen_api_key  # 设置 Qwen API key
        openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 设置 Qwen API base URL
        
        # 添加随机种子到提示词中，增加响应的多样性
        seed = random.randint(1, 1000000)
        modified_prompt = f"{prompt}\n\n[随机种子: {seed}]"
        
        # 限制发送给 API 的图片数量，减少内存使用和请求大小
        max_images = 5
        selected_frames = base64Frames[:max_images]
        print(f"Using {len(selected_frames)} images out of {len(base64Frames)} total frames")
        
        # 构建消息结构
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": modified_prompt
                    },
                    {
                        "type": "text",
                        "text": weather
                    }
                ]
            }
        ]

        # 添加图片到消息中
        for frame in selected_frames:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })

        print("Sending request to Qwen API...")
        
        # 使用旧版本的 API 调用格式
        response = openai.ChatCompletion.create(
            model="qwen-vl-max-latest",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.6,
            frequency_penalty=0.6,
            seed=seed
        )
        
        print("Received response from Qwen API")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in Qwen API call: {str(e)}")
        raise

@app.route('/')
def index():
    try:
        # 添加调试信息
        print(f"Current working directory: {os.getcwd()}")
        print(f"Template folder: {app.template_folder}")
        print(f"Available templates: {os.listdir(app.template_folder) if os.path.exists(app.template_folder) else 'No template folder found'}")
        
        return render_template('index.html', default_prompt=DEFAULT_PROMPT)
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        print(f"Template folder: {app.template_folder}")
        print(f"Current working directory: {os.getcwd()}")
        # 返回一个简单的错误页面
        return f"""
        <html>
            <body>
                <h1>Error</h1>
                <p>Failed to load template: {str(e)}</p>
                <p>Template folder: {app.template_folder}</p>
                <p>Current directory: {os.getcwd()}</p>
            </body>
        </html>
        """, 500

@app.route('/process', methods=['POST'])
def process_video():
    if 'folder' not in request.files:
        return jsonify({'error': '没有上传文件夹'}), 400
    
    # 获取上传的文件夹
    folder = request.files.getlist('folder')
    if not folder:
        return jsonify({'error': '文件夹为空'}), 400

    # 获取提示词和模型选择
    prompt = request.form.get('prompt', DEFAULT_PROMPT)
    weather = request.form.get('weather', "")
    model = request.form.get('model', 'qwen-vl-max-latest')
    
    # 添加调试信息
    print(f"Selected model: {model}")
    print(f"Available form data: {request.form}")

    # 创建临时文件夹存储图片
    temp_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_folder, exist_ok=True)

    try:
        # 保存所有图片
        for file in folder:
            if file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(temp_folder, filename))

        # 获取所有图片文件并按名称排序
        image_files = sorted(glob.glob(os.path.join(temp_folder, "*.png")))
        
        # 只保留每25帧中的一帧
        selected_frames = image_files[::25]
        print(f"Total frames: {len(image_files)}, Selected frames: {len(selected_frames)}")

        # 处理选中的图片序列
        base64Frames = []
        for image_path in selected_frames:
            try:
                # 使用 cv2.IMREAD_REDUCED_COLOR_2 来降低图片质量
                frame = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)
                if frame is None:
                    continue
                # 使用较低的 JPEG 质量
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                # 立即释放内存
                del frame
                del buffer
            except Exception as e:
                print(f"Error processing frame {image_path}: {str(e)}")
                continue

        # 清理不需要的图片文件
        for file in image_files:
            if file not in selected_frames:
                try:
                    os.remove(file)
                except:
                    pass

        # 根据选择的模型处理请求
        print(f"Processing with model: {model}")
        if model == "gpt-4.1-mini":  # 使用原来的模型名称
            print("Using OpenAI API")
            commentary = process_with_openai(prompt, base64Frames)
        else:  # qwen-vl-max-latest
            print("Using Qwen API")
            commentary = process_with_qwen(prompt, weather, base64Frames)

        # 清理临时文件
        shutil.rmtree(temp_folder)

        return jsonify({
            'success': True,
            'commentary': commentary,
            'model_used': model,
            'timestamp': datetime.datetime.now().isoformat()
        })

    except Exception as e:
        # 确保清理临时文件
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        print(f"Error in process_video: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 在生产环境中使用环境变量中的端口，默认为 8080
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 