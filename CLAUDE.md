# AnnaVoices 开发规则

## 测试

- **提交前必须在本地跑通测试。** 只有全部通过才能提交：

  ```bash
  .venv\Scripts\python -m pip install -r requirements-test.txt
  .venv\Scripts\python -m pytest
  ```

- **新功能必须附带测试。** 修 bug 时先写一条能复现问题的测试，再改代码。
- 测试不得下载模型、联网、调用 ffmpeg 或依赖 GPU。`tests/conftest.py` 默认拦截这些操作，需要用桩函数代替。
- 修好标记为 `xfail` 的已知问题后，要删掉对应标记。`strict` 模式下，不删标记测试会失败。

## 提交流程

- `main` 受保护，所有改动都走分支和 PR，CI 通过才能合并。
- 新增测试依赖时写进 `requirements-test.txt`，与 `requirements.txt` 共有的包版本要保持一致。
