import cosysairsim as airsim
import time

client = airsim.MultirotorClient(ip="172.19.160.1")
client.confirmConnection()

# 車輛
car_names = [name for name in client.simListSceneObjects() if
             "vehCar" in name or "vehVan" in name or "vehTruck" in name or "Vehicle" in name]

# 人物
human_names = [name for name in client.simListSceneObjects() if "CrowdCharacter" in name]

# 記錄上次位置
last_car_poses = {}
last_human_poses = {}

while True:
    # 追蹤車輛
    for name in car_names:
        pose = client.simGetObjectPose(name)
        pos = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
        last = last_car_poses.get(name)
        if last and (abs(pos[0] - last[0]) > 0.1 or abs(pos[1] - last[1]) > 0.1 or abs(pos[2] - last[2]) > 0.1):
            print(f"🚗 {name} 移動到 ({pos[0]:.1f}, {pos[1]:.1f})")
        last_car_poses[name] = pos

    # 追蹤人物
    for name in human_names:
        pose = client.simGetObjectPose(name)
        pos = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
        last = last_human_poses.get(name)
        if last and (abs(pos[0] - last[0]) > 0.1 or abs(pos[1] - last[1]) > 0.1 or abs(pos[2] - last[2]) > 0.1):
            print(f"🧑 {name} 移動到 ({pos[0]:.1f}, {pos[1]:.1f})")
        last_human_poses[name] = pos

    time.sleep(1)
