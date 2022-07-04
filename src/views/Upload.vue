<template>
  <div class="container">
    <el-row>
      <el-col :span="12">
        <el-form :form="uploadForm" label-width="100px">
          <el-form-item label="菜品名称">
            <el-input v-model.trim="uploadForm.foodName"></el-input>
          </el-form-item>
          <el-form-item label="菜品价格">
            <el-input v-model.trim="uploadForm.price"></el-input>
          </el-form-item>
          <el-form-item label="上传图片">
            <el-upload
              action="#"
              list-type="picture-card"
              accept="image/*"
              :limit="10"
              :multiple="true"
              :auto-upload="false"
              :file-list="uploadForm.fileList"
              :on-change="pictureChangeHandler"
              :on-remove="pictureRemoveHandler"
            >
              <el-icon>
                <Plus />
              </el-icon>
            </el-upload>
          </el-form-item>
          <el-form-item label>
            <el-button size="small" type="primary" @click="onSubmit">确定</el-button>
            <el-button size="small" @click="onCancel">取消</el-button>
          </el-form-item>
        </el-form>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'

import { ElMessage } from 'element-plus'

import request from '../utils/request'
import api from '../utils/api'

const uploadForm = reactive({
  foodName: '',
  price: '',
  fileList: []
})
const pictureChangeHandler = (file, fileList) => {
  uploadForm.fileList = fileList
}
const pictureRemoveHandler = (file, fileList) => {
  uploadForm.fileList = fileList
}
// 上传菜品
const id = ref(0) // 做菜品-图片校验
const uploadDish = async (foodName, price) => {
  const dish = await request({
    url: api.ADD_DISH,
    method: 'POST',
    data: {
      dish_name: foodName,
      dish_price: price
    }
  })
  id.value = dish.dish_id
}
// 上传菜品-图片
const uploadPics = async (id, file) => {
  const data = new FormData()
  data.append('dish_id', `${id}`)
  data.append('file', file.raw)
  await request({
    url: api.ADD_PIC,
    method: 'POST',
    headers: {
      'content-type': 'multipart/form-data'
    },
    data
  })
}
// 上传
const onSubmit = async () => {
  const { foodName, price, fileList } = uploadForm
  if (verifyUploadForm({ foodName, price, fileList })) {
    try {
      await uploadDish(foodName, price)
      const uploadPicsList = fileList.map((file) => uploadPics(id.value, file))
      console.log('uploadPicsList', uploadPicsList)
      await Promise.all(uploadPicsList)
      ElMessage.success('上传成功')
      router.push('/')
    } catch (e) {
      ElMessage.error('菜品上传错误,请稍后重试')
      console.log('e', e)
    }
  }
}
// 取消上传
const router = useRouter()
const onCancel = () => router.push('/')
// 校验 form
const verifyUploadForm = ({ foodName, price, fileList }) => {
  if (!foodName) {
    ElMessage.warning('请填写菜品名称')
    return false
  }
  if (!price) {
    ElMessage.warning('请填写菜品价格')
    return false
  }
  if (isNaN(+price)) {
    ElMessage.warning('菜品价格应为数字')
    return false
  }
  if (!fileList.length) {
    ElMessage.warning('请选择图片后再上传')
    return false
  }
  return true
}
</script>

<style scoped>
</style>