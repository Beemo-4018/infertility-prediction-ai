import os

def get_data_path(filename):
    # 1. DACON 서버 환경인 경우
    dacon_path = f"/data/{filename}"
    if os.path.exists("/data"):
        return dacon_path
    
    # 2. 로컬 환경인 경우 (상대 경로들을 다 시도해봄)
    paths = [
        os.path.join("data", "raw", filename),      # main.py에서 부를 때
        os.path.join("..", "data", "raw", filename) # notebooks/에서 부를 때
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
            
    # 파일을 못 찾은 경우 기본값 반환
    return paths[0]