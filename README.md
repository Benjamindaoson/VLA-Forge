# VLA-Forge

**�����˲���ʧ�ָܻ������ž����������Ч VLA ����**
*Failure recovery, replay-verified corrective data, and efficient VLA inference.*

VLA-Forge ����������ѧϰ�ŶӰ�һ�β���ʧ�ܱ�ɿ�׷�ݡ����طŵ�������������������ʧ����Դ���ָ� MuJoCo ����״̬��ִ���ܿظ�Ԥ���µĲ��Զ�������֤�ɹ��źţ��ٶ����طŲ�д�� LeRobotDataset v3��

**��ǰ�����2026-10-11����** M04 ����� 11 ���¾����켣�Ķ����ط���֤���Լ� FP32 CUDA Graph �����Ż������ݼ�����ʾ��˽�н�������������������Ϊ **INCOMPLETE**���ȴ���ȷ�زİ汾���ٷַ�����ȷ�ϡ�M05 ģ��ѵ����δ��չ��

## ��ʾ����Ƶ�빤������

| ���� | ��� | �������� |
| --- | --- | --- |
| ������Ŀ Demo | [���� / ���� 227.934 �� MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/M04_demo_200s.mp4) | ˽�У���Ҫ HF �ֿ����Ȩ�� |
| ʵ��ͼ�� | [Hugging Face Space](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence) | ˽�У���Ҫ HF �ֿ����Ȩ�� |
| 37 ����ʵʵ���ز� | [������Ƶ������������������](docs/results/M04/VIDEO_INDEX.md) | ������������Ƶ˽�� |
| ��Ŀ������ͼ | [M04 overview](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/M04_overview.png) | ˽�� |
| ��Ԥ���ս�ͼ | [M04 comparison](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/M04_comparison.png) | ˽�� |
| ���Ĺ������� | [�ɹ�����֤���ɱ�������](docs/results/M04/WORK_REPORT.md) | ���ֿ�ɶ� |
| ԭʼ����� | [CLOSEOUT_REPORT](docs/results/M04/CLOSEOUT_REPORT.md) | ���ֿ�ɶ� |
| �����ɶ���� | [���ժҪ��Դ�ļ�У����Ϣ](docs/results/M04/results_summary.json) | ���ֿ�ɶ� |
| LeRobotDataset v3 | [Corrections development dataset](https://huggingface.co/datasets/jlai300/vla-forge-m04-corrections-dev) | ˽�У���Ҫ HF �ֿ����Ȩ�� |

��Ƶ���**1920 �� 1080��H.264��yuv420p��30 FPS��43,383,032 bytes**��������Ƶ��ͨ������� 37 ����Դ��Ƶ�Ĺ�ϣ��֤�������ɹ���ʧ�ܰ��������л����˻���������ʵ LIBERO/MuJoCo ���档

˽�����Ӳ����ڹ���������Ҳ����������� README �ж��ı����Ȩ�ޡ�ͼ�����Ƶ���ƶ�ԭʼ�ļ����ֲ��䡣

## ʵ�ʹ�������

```text
M02 ��ʵʧ�ܹ켣
  �� ��Դ Episode / ���� / ״̬ / Hash
  �� �ָ��ο�����״̬����֤һ����
  �� �ܿظ�Ԥ + ���� SmolVLA �����µĶ���
  �� MuJoCo ִ�� + Success �ź�
  �� ���������ط�����
  �� ȥ�ء���Դ�����붯��/�۲����
  �� LeRobotDataset v3 + ��Ƶ + ʵ���ִ
```

����ջ���ٷ� `lerobot/smolvla_libero`��LeRobot��LIBERO / robosuite / MuJoCo��PyTorch��EGL ˫���������Ⱦ��RTX 3090 24 GB�����ö����ģ�͡�Ԥ�����붯��Э�飬û��ͨ������ģ�������ͱ��μ��١�

## �������ݣ���Щ�Ѿ���֤

| ��Ŀ | ʵ���� |
| --- | --- |
| �²�����ȥ�ص� M04 �������� | **11 ��** |
| ���������ط� | ÿ���¾��� **2 ��ͨ��** |
| ���б�ǵ���ʷ�������� | 1 �� M03 �����켣 |
| LeRobotDataset v3 r7 | **12 Episodes / 956 Transitions** |
| ������Դ��Χ | LIBERO Goal task 6��M02 init states 6��7 |
| д��������ȡ�Ķ��� / ״̬������ | **0 / 0** |
| ͼ������ƽ��������� | 2.190������ֵ��λ�����Զ������ݼ���ƣ� |
| ���ݱ������� | 26.21 Transitions/s��д�� 36.478 s |
| ȫ���������� | 79 ���ǼǼ�¼��12 �� Success ��¼��67 ��ʧ�ܼ�¼ |
| ��֤��ʧ�ܱ��� | ʧ����־��ԭʼ�켣����Ƶ����Դ��ִ���� |

**12 �� Success �ǼǼ�¼������ 12 ���µĶ���������** ����������ȥ�ز�������֤���� M04 ����Ϊ 11 �������ݼ��ĵ� 12 �����Ե�����ǵ� M03 �����켣��79 ���������ڿ����������̣�������Ϊ��׼���Լ��ɹ��ʡ�

�ﵽ���� 10 ���¾���������Ҫ����δ�ﵽ 30�C50 ����չĿ�ꡣ��ǰ������һ������������Դ��ʼ״̬��**��֧�ֿ����񷺻���ģ��ѵ����������ʵ�����˲���Ч���Ľ���**���������������ڿ�������Դ����������Ӧ�����ų������� M06 �������Լ������׶�û��ִ�� M06��

## �������٣����� FP32 �������������

18 ����ԱȽ����� **3 ����ʵ�۲� �� 6 ����������**����ʱ���� GPU ͬ���������ԭʼ�����飬����ǰ�������������档

| ָ�� | Eager FP32 | CUDA Graph FP32 |
| --- | ---: | ---: |
| P50 | 391.74 ms | **56.39 ms** |
| P95 | 398.84 ms | **61.85 ms** |
| P99 | 399.07 ms | **61.87 ms** |
| Ԥ�Ⱥ� P50 ���ٱ� | 1.00�� | **6.95��** |
| 18 �鶯���������Բ��� | �� | **0** |
| 18 �������״̬һ�� | �� | **ȫ��һ��** |

��̬�Ӿ�ǰ׺���� eager ·��������ɸ��õ�ע������ȥ����㡣����ģ��Ȩ�ء�FP32 ���ȡ�ȥ�벽��Ͷ��������ò��䡣BF16 / TF32 ��̽�����δ��Ϊͨ����ȷ�����յ��Ż�������

**ʵ������ִ����֤��** ��ͬһ������Դ�� MuJoCo �����У�ԭʼ�汾�� CUDA Graph �汾���� **21 ������������ Success=True**���������ؽ�λ�����ٶȡ�ĩ��λ�ˡ�˫���֡���������Ϊ 0��[�������ջ�ִ](docs/results/M04/cuda_graph_physical_comparison.json)

**�����������÷�Χ��** �״�ͼ����Լ **1.82 s**�����ζ����ܵ�����������ѭ��Ϊ 6.965 s����ʷ eager �ο�Ϊ 5.558 s���ⲻ��ͬʱִ�еĶ˵������ܶ��գ�Ҳδ֤���˵��˼��١�Ĭ�ϱ��� eager��CUDA Graph �����ܹ��������� sampler �ĳ�����Լ 6 ���������̯������ֻ�����ڱ���΢��׼��[�������ܻ�ִ](docs/results/M04/cuda_graph_benchmark.json)

## ��Դ���������뵱ǰ����

- �ѵǼǵ���������ѭ���ۼ� **0.573466 GPU-hours**��ÿ���¿��ž���Լ **0.052133 GPU-hours**������ʧ�ܳ��ԣ����������������ء�ǰ׺�ָ��������طš�����Ϳ���ʱ�䣬**�������������ɱ������˵�**��
- ��������ѭ����ֵ PyTorch allocated memory Ϊ 0.898 GiB��CUDA Graph ΢��׼��ֵΪ 981,761,024 bytes�����߶������ڽ������Դ档
- 237 ���Ǽ���Ʒ���������ļ�ȱʧ���� **1 ����ʷʧ�ܳ��Ե� JSON ��ϣ�쳣**�������䶯���������Ӧ NPZ ��ͬ������ϣ�쳣��Ȼ���ڡ�
- �ƶ˽�����֤��**82 passed��1 ������ Starlette ���þ���**��Ruff �������ͨ���������ƶ�Դ��汾����֤��¼������ð�乫�� main ��֧ CI��
- ���ξ�ȷ�زİ汾���ٷַ������Դ�ȷ�ϡ���������֤�����Զ�����ȫ��ģ�͡����ݡ������������Ȩ���˲顣
- û�п�ʼ M05 ѵ����û�����ƾ��������Ѿ�����ģ�ͳɹ���������

## ���������汾

Ȩ���з�Ŀ¼Ϊ�ƶ� `/data/VLA-Forge/project/robot-vla`������֤��Ϊ `/data/VLA-Forge/Evidence/M04`������ģ�͡���Ƶ��켣λ���ⲿ��ƷĿ¼��û�м��� Git��

```bash
# �������õ���Ȩ RTX 3090 �ƻ���������
bash /data/VLA-Forge/Evidence/M04/reproduce.sh audit
bash /data/VLA-Forge/Evidence/M04/reproduce.sh benchmark

# ��ѡ���½�һ��������֤���ܣ�����ʹ��δ�ù��� ID
M04_RUN_ID=my_unique_graph_check \
  bash /data/VLA-Forge/Evidence/M04/reproduce.sh rollout
```

- ����֤���㷨Դ���ύ��`1836156f8f4ef65014237fed5c6a66f00d23d076`��
- �����ĵ�����ǰ���ƶ�Դ��汾��`0dc9a02ab601235e2571dc9ab37800a8b1105f8f`��
- �����汾�����ƶ� `codex/m04-verified-corrections-3090` ��֧��M04 Դ����ʷ��δͬ���� GitHub�����η��������ĵ����ı�֤�ݣ����ܰѹ��� main �����Ѱ���ȫ�� M04 ʵ����롣
- �ѷ����� M02 Դ�룺[codex/m02-libero40-benchmark](https://github.com/Benjamindaoson/VLA-Forge/tree/codex/m02-libero40-benchmark)��
- [��ȷ�����汾](docs/results/M04/dependency_versions.json)��[���ֽű�����](docs/results/M04/reproduce.sh)��[ʵ���������Դ��ϣ](docs/results/M04/results_summary.json)��

## ���̽ṹ������ԭ��

`src/vla_forge/` �ṩʧ�ܰ�������ԤЭ�顢����������֤����Դ���顢�洢���о� API���ƶ� `benchmarks/robot-vla-engine/` ������ʵ SmolVLA/LIBERO Runner���ط���֤�����ݵ���������ʵ�顣

���������ڹ���ԭ�ͣ�

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,serve]"
.venv/bin/python -m pytest tests -q
.venv/bin/vla-forge --help
```

����ԭ�͡�CLI ����������ϳ�ʾ����������������ʵʵ�顣Windows ��ʹ�� `.venv\Scripts\python.exe`����ʽ GPU ʵ�������ƶ����С�

- [�ܹ�˵��](docs/ARCHITECTURE.md) �� [ʵ��Э��](docs/EXPERIMENT_PROTOCOL.md) �� [��ȫ�߽�](docs/SAFETY.md)
- [README ����ǰ����ʷ����](docs/history/README_before_M04_showcase_20261011.md)����������״̬���Ѳ�������ǰ�ƶ�ʵ����ȡ�

<details>
<summary>չ��ȫ�� 37 ����ʵʵ����Ƶ��������˽���زģ�</summary>

| ��Դ Episode | Suite / Task | ��� | ʵ������ | ��Ƶ |
| --- | --- | --- | --- | --- |
| M02-spatial-t00-e01 | spatial / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/01_M02-spatial-t00-e01.mp4) |
| M02-spatial-t00-e02 | spatial / 0 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/02_M02-spatial-t00-e02.mp4) |
| M02-object-t00-e00 | object / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/03_M02-object-t00-e00.mp4) |
| M02-object-t00-e03 | object / 0 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/04_M02-object-t00-e03.mp4) |
| M02-goal-t00-e00 | goal / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/05_M02-goal-t00-e00.mp4) |
| M02-goal-t02-e03 | goal / 2 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/06_M02-goal-t02-e03.mp4) |
| M02-long-t00-e01 | long / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/07_M02-long-t00-e01.mp4) |
| M02-long-t00-e00 | long / 0 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/08_M02-long-t00-e00.mp4) |
| m03-g09-goal-t06-init07_baseline | goal / 6 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/09_m03-g09-goal-t06-init07_baseline.mp4) |
| m03-g09-goal-t06-init07_brightness | goal / 6 | Success | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/10_m03-g09-goal-t06-init07_brightness.mp4) |
| m03-g01-spatial-t05-init00_baseline | spatial / 5 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/11_m03-g01-spatial-t05-init00_baseline.mp4) |
| m03-g01-spatial-t05-init00_brightness | spatial / 5 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/12_m03-g01-spatial-t05-init00_brightness.mp4) |
| m03-g02-spatial-t05-init01_baseline | spatial / 5 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/13_m03-g02-spatial-t05-init01_baseline.mp4) |
| m03-g02-spatial-t05-init01_occlusion | spatial / 5 | Failure | occlusion | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/14_m03-g02-spatial-t05-init01_occlusion.mp4) |
| m03-g03-long-t00-init00_baseline | long / 0 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/15_m03-g03-long-t00-init00_baseline.mp4) |
| m03-g03-long-t00-init00_brightness | long / 0 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/16_m03-g03-long-t00-init00_brightness.mp4) |
| m03-g04-long-t00-init02_baseline | long / 0 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/17_m03-g04-long-t00-init02_baseline.mp4) |
| m03-g04-long-t00-init02_occlusion | long / 0 | Failure | occlusion | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/18_m03-g04-long-t00-init02_occlusion.mp4) |
| m03-g05-long-t04-init00_baseline | long / 4 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/19_m03-g05-long-t04-init00_baseline.mp4) |
| m03-g05-long-t04-init00_brightness | long / 4 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/20_m03-g05-long-t04-init00_brightness.mp4) |
| m03-g06-long-t04-init01_baseline | long / 4 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/21_m03-g06-long-t04-init01_baseline.mp4) |
| m03-g06-long-t04-init01_occlusion | long / 4 | Failure | occlusion | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/22_m03-g06-long-t04-init01_occlusion.mp4) |
| m03-g07-goal-t04-init04_baseline | goal / 4 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/23_m03-g07-goal-t04-init04_baseline.mp4) |
| m03-g07-goal-t04-init04_brightness | goal / 4 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/24_m03-g07-goal-t04-init04_brightness.mp4) |
| m04-diverse-goal-t06-init06-seed84182 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/25_m04-diverse-goal-t06-init06-seed84182.mp4) |
| m04-goal-t06-e07-brightness115-first50-seed84075 | libero_goal / 6 | Success | input_brightness_gain_1.15_first50 | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/26_m04-goal-t06-e07-brightness115-first50-seed84075.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84068 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/27_m04-goal-t06-e07-wrist110-first50-seed84068.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84071 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/28_m04-goal-t06-e07-wrist110-first50-seed84071.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84072 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/29_m04-goal-t06-e07-wrist110-first50-seed84072.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84088 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/30_m04-goal-t06-e07-wrist110-first50-seed84088.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84092 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/31_m04-goal-t06-e07-wrist110-first50-seed84092.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84093 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/32_m04-goal-t06-e07-wrist110-first50-seed84093.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84094 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/33_m04-goal-t06-e07-wrist110-first50-seed84094.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84095 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/34_m04-goal-t06-e07-wrist110-first50-seed84095.mp4) |
| m04-goal-t06-e07-wrist115-first50-seed84067 | goal / 6 | Success | input_brightness_gain_1.15_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/35_m04-goal-t06-e07-wrist115-first50-seed84067.mp4) |
| m04-spatial-t05-e07-brightness115-first50-seed82057 | spatial / 5 | Failure | input_brightness_gain_1.15_steps0-50 | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/36_m04-spatial-t05-e07-brightness115-first50-seed82057.mp4) |
| m04-long-t04-e04-brightness115-first50-seed85044 | long / 4 | Failure | input_brightness_gain_1.15_steps0-50 | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/37_m04-long-t04-e04-brightness115-first50-seed85044.mp4) |

</details>
