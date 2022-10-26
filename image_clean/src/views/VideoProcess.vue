<template>
  <div>
    <div class="container">
      <el-form :model="form" label-width="160px" label-position="left">
        <el-form-item label="视频文件列表">
          <el-select size="large" v-model="data.path" placeholder="请选择要处理的视频" style="width: 500px;">
            <el-option v-for="path in data.allVideoPath" :label="getNameFromPath(path)" :value="path" />
          </el-select>
        </el-form-item>

        <el-form-item label="FPS(每秒采样次数)">
          <el-radio-group v-model="data.fps">
            <el-radio v-for="i in fps_options" :label="i"></el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item label="画面差异阈值">
          <el-slider v-model="data.diff_area" :min="0.05" :max="0.75" :step="0.05" />
          <label>{{ data.diff_area }}</label>
        </el-form-item>

        <el-form-item label="画面稳定时间(毫秒)">
          <el-slider v-model="data.stable" :min="100" :max="3000" :step="50" />
          <label>{{ data.stable }}</label>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="onSubmit">开始处理</el-button>
        </el-form-item>
      </el-form>
    </div>

  </div>
</template>

<script setup>
import {reactive, onMounted} from 'vue'
import request from '../utils/request'
import ElMessage from 'element-plus'

const fps_options = [5, 10, 15, 25, 30]

const data = reactive({
  allVideoPath: [], // 获取视频文件列表
  path: '',
  fps: 15,
  diff_area: 0.2,
  stable: 900,
})

onMounted(async () => {
    const res = await request({
        url: '/get_video_paths',
        method: 'get'
    })
    data.allVideoPath = res.data
})
const getNameFromPath = (path) =>{
  return path.substr(path.lastIndexOf('/')+1)
}

const onSubmit = async () => {
  if (data.path.length == 0) {
    ElMessage({
      showClose: true,
      message: '请选择一个需要处理的视频',
      type: 'warning',
    })
  } else {
    const res = await request({
      url: '/process_video',
      method: 'GET',
      params: {
        'path': data.path,
        'fps': data.fps,
        'diff_area': data.diff_area,
        'stable': data.stable
      }
    })
    ElMessage({
      showClose: true,
      message: '处理完成',
      type: 'success',
    })

  }
}
</script>