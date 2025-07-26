# AirSim 無人機控制與視覺系統 (AirSim Drone Control & Vision System)

這是一個基於 AirSim 模擬環境的先進無人機控制與電腦視覺開發平台。此系統整合了 MAVSDK、ROS2 和一個使用 Tkinter 的圖形化使用者介面 (GUI)，提供了一個用於開發、測試和驗證無人機自主飛行與視覺演算法的完整解決方案。

## ✨ 主要功能

- **分離式架構**: 控制系統 (`control`) 與視覺顯示系統 (`display`) 可以同時運行，也可以獨立運行，提供高度的靈活性。
- **手動飛行控制**:
    - 提供一個功能完整的 Tkinter GUI，可即時監控無人機狀態。
    - 支援鍵盤對無人機進行精確的姿態、速度和偏航控制。
    - 控制指令經過平滑處理，提供流暢的飛行體驗。
- **多相機視覺串流**:
    - 可同時顯示來自 AirSim 的多個相機視角（例如：前置、左側、右側）。
    - 顯示視窗中疊加了即時的遙測數據，如高度、姿態和 FPS。
- **雲台控制 (Gimbal Control)**:
    - 支援透過鍵盤獨立控制相機雲台的俯仰、滾轉和偏航。
- **MAVSDK 整合**: 透過 MAVSDK 與 PX4 等飛控軟體進行通訊，執行起飛、降落、切換至 Offboard 模式等指令。
- **ROS2 整合**:
    - 包含一個 ROS2 橋接節點 (`airsim_bridge.py`)，可將 AirSim 的相機影像和 IMU 數據發佈到 ROS2 主題 (topics) 上。
- **立體視覺支援**:
    - 內建一個立體影像檢視器 (`viewer.py`)。
    - 自動將影像和 IMU 數據儲存為 [OpenVINS](https://github.com/rpng/open_vins) 相容的格式，方便進行視覺慣性里程計 (VIO) 的開發與測試。

## 📂 專案結構

```
.
├── src/
│   ├── configs/
│   │   └── airsim_config.py      # Pydantic 模型，用於所有組態設定
│   ├── task/
│   │   ├── control.py            # 無人機控制邏輯和 GUI
│   │   └── display.py            # AirSim 影像顯示和處理
│   ├── utils/
│   │   ├── logger.py             # 自訂日誌記錄器
│   │   └── util.py               # 事件總線 (EventBus) 和其他工具
│   ├── stereo_images/
│   │   ├── airsim_bridge.py      # ROS2 橋接 (立體相機版本)
│   │   └── viewer.py             # 立體影像檢視器與 OpenVINS 數據產生器
│   ├── unrealengine/
│   │   └── catch_pos.py          # 用於追蹤 Unreal Engine 場景中物件位置的腳本
│   ├── airsim_bridge.py          # ROS2 橋接 (單相機版本)
│   └── main.py                   # 系統主入口點
├── .gitignore
└── README.md
```

## 🚀 環境設定與安裝

1.  **複製專案**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **設定 AirSim 環境**:
    - 請先安裝 Unreal Engine。
    - 依照 [AirSim 官方文件](https://microsoft.github.io/AirSim/) 的指示設定一個 AirSim 模擬環境（例如：CityEnviron 或 Blocks）。
    - 在 AirSim 的 `settings.json` 中，請確保已設定 `PX4` 模式並啟用 API 控制。

3.  **設定 Python 環境**:
    - 建議使用虛擬環境：
      ```bash
      python -m venv env
      source env/bin/activate  # On Windows, use `env\Scripts\activate`
      ```
    - 建立 `requirements.txt` 檔案，並包含以下主要依賴套件：
      ```txt
      numpy
      opencv-python
      pydantic
      cosysairsim  # or airsim
      mavsdk
      # 如果需要使用 ROS2 功能，請確保已安裝 ROS2 環境
      ```
    - 安裝依賴：
      ```bash
      pip install -r requirements.txt
      ```

## 🕹️ 如何使用

### 整合式系統 (控制 + 顯示)

執行主腳本 `main.py`。系統會同時啟動控制 GUI 和相機顯示視窗。

```bash
python src/main.py --ip <your_airsim_ip>
```

**常用參數**:
- `--display-only`: 只啟動顯示視窗。
- `--control-only`: 只啟動控制 GUI。
- `--ip`: 指定 AirSim 伺服器的 IP 位址。
- `--mavsdk-port`: 指定 MAVSDK 的 UDP 連接埠。
- `--verbose`: 啟用詳細的日誌輸出。
- `--width`, `--height`: 設定從 AirSim 請求的影像解析度。
- `--output-width`, `--output-height`: 設定顯示視窗的影像解析度。

### 獨立模組

- **立體影像檢視器 (Stereo Viewer)**:
  ```bash
  python src/stereo_images/viewer.py
  ```
  此腳本會連接到 AirSim，顯示左右相機的畫面，並將數據儲存在 `openvins_data` 資料夾中。

- **ROS2 橋接 (ROS2 Bridge)**:
  - 確保您的 ROS2 環境已設定。
  - 執行橋接節點：
    ```bash
    ros2 run <your_package_name> airsim_bridge
    ```

## ⌨️ 鍵盤控制

| 按鍵                  | 功能                       |
| --------------------- | -------------------------- |
| **W / S** | 前進 / 後退                |
| **A / D** | 向左 / 向右                |
| **R / F** | 上升 / 下降                |
| **Q / E** | 向左偏航 / 向右偏航        |
| **I / K** | 雲台俯仰 (上 / 下)         |
| **J / L** | 雲台偏航 (左 / 右)         |
| **U / O** | 雲台滾轉 (左 / 右)         |
| **Z / X** | 增加 / 減少速度倍率        |
| **C** | 重設速度倍率為 1.0x        |
| **ESC** | **緊急停止並降落** |

## ⚙️ 組態設定

所有系統參數，包括連線設定、相機參數、控制靈敏度等，都可以在 `src/configs/airsim_config.py` 中進行修改。

## 📜 授權

MIT License。