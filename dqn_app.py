import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces

# -------------------------------
# 1. Custom Environment
# -------------------------------

class WorkoutSchedulerEnvV2(gym.Env):
    def __init__(self):
        super(WorkoutSchedulerEnvV2, self).__init__()
        self.observation_space = spaces.Box(
            low=np.array([10, 0, 120, 30, 0, 0, 0, 4, 0, 10, 0, 0], dtype=np.float32),
            high=np.array([80, 1, 220, 150, 4, 180, 1, 12, 1, 40, 2, 1], dtype=np.float32)
        )
        self.action_space = spaces.Discrete(60)
        self.activity_map = {
            0: "Jalan kaki",
            1: "Jogging",
            2: "Yoga",
            3: "Senam ringan",
            4: "Sepeda statis"
        }

    def decode_action(self, action):
        waktu = action // 20
        jenis = (action % 20) // 4
        durasi = (action % 20) % 4
        return waktu, jenis, durasi

# -------------------------------
# 2. Model DQN
# -------------------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# === DQN MODEL ===
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# === REPLAY BUFFER (opsional kalau mau latihan batch) ===
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), actions, rewards, np.array(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# === INISIALISASI ENV, MODEL, DEVICE ===
env = WorkoutSchedulerEnvV2()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

# -------------------------------
# 3. Streamlit Interface
# -------------------------------

st.set_page_config(page_title="Rekomendasi Olahraga Adaptif", layout="centered")
st.title("🏃‍♀️ Sistem Rekomendasi Jadwal Olahraga Adaptif")
st.markdown("Menggunakan **Deep Q-Network (DQN)** untuk merekomendasikan waktu, jenis, dan durasi olahraga berdasarkan kondisi pengguna dan mode puasa.")

# Inisialisasi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = WorkoutSchedulerEnvV2()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load model DQN
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("dqn_model.pth", map_location=device))
model.eval()

# Mapping untuk decoding
waktu_map = {0: "Pagi", 1: "Sore", 2: "Tidak olahraga"}
jenis_map = {
    0: "Jalan kaki",
    1: "Jogging",
    2: "Yoga",
    3: "Senam ringan",
    4: "Sepeda statis"
}
durasi_map = {
    0: "15 menit",
    1: "30 menit",
    2: "45 menit",
    3: "60 menit"
}

# -------------------------------
# 4. Form Input
# -------------------------------

with st.form("input_form"):
    st.subheader("📋 Masukkan Data Pengguna:")
    usia = st.slider("Usia", 10, 80, 25)
    gender = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    tinggi = st.slider("Tinggi Badan (cm)", 130.0, 200.0, 165.0)
    berat = st.slider("Berat Badan (kg)", 30.0, 150.0, 60.0)
    aktivitas = st.selectbox("Jenis Aktivitas Terakhir", ["Santai", "Ringan", "Sedang", "Tinggi", "Sangat Berat"])
    durasi = st.slider("Durasi Olahraga Terakhir (menit)", 0.0, 120.0, 30.0)
    intensitas = st.slider("Intensitas Olahraga (0 - 1)", 0.0, 1.0, 0.5)
    tidur = st.slider("Jam Tidur Terakhir", 4.0, 10.0, 7.0)
    stres = st.slider("Tingkat Stres (0 - 1)", 0.0, 1.0, 0.3)
    kesehatan = st.selectbox("Kondisi Kesehatan", ["Baik", "Sedang", "Lemah"])
    mode = st.radio("Mode", ["Normal", "Puasa"])

    submitted = st.form_submit_button("Dapatkan Rekomendasi")

if submitted:
    # Encoding input ke dalam state vektor
    gender_val = 1 if gender == "Laki-laki" else 0
    aktivitas_val = ["Santai", "Ringan", "Sedang", "Tinggi", "Sangat Berat"].index(aktivitas)
    kesehatan_val = ["Baik", "Sedang", "Lemah"].index(kesehatan)
    mode_val = 0 if mode == "Normal" else 1
    bmi = berat / ((tinggi / 100) ** 2)

    state = np.array([
        usia, gender_val, tinggi, berat, aktivitas_val, durasi,
        intensitas, tidur, stres, bmi, kesehatan_val, mode_val
    ], dtype=np.float32)

    # Prediksi aksi dari model
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    waktu, jenis, durasi_kode = env.decode_action(action)

    st.success("✅ Rekomendasi Olahraga dari Model")
    st.markdown(f"""
    - 🕒 **Waktu Olahraga:** {waktu_map[waktu]}  
    - 🏃 **Jenis Olahraga:** {jenis_map.get(jenis, 'Tidak diketahui')}  
    - ⏱️ **Durasi:** {durasi_map[durasi_kode]}  
    """)

    with st.expander("🔍 Lihat Q-Values"):
        st.write(q_values.cpu().numpy())

