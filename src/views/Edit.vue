<template>
  <div class="container">
    <el-row>
      <el-col :span="12">
        <el-form :form="form" label-width="100px">
          <el-form-item label="菜品名称:" prop="name">
            <el-input v-model="form.name"></el-input>
          </el-form-item>
          <el-form-item label="菜品价格:" prop="price">
            <el-input v-model="form.price"></el-input>
          </el-form-item>
          <el-form-item label="菜品图片:">
            <el-upload
              action="#"
              list-type="picture-card"
              accept="image/*"
              :multiple="true"
              :auto-upload="false"
              :file-list="form.fileList"
              :on-change="pictureChangeHandler"
              :on-preview="picturePreviewHandler"
              :on-remove="pictureRemoveHandler"
            >
              <el-icon>
                <Plus />
              </el-icon>
            </el-upload>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="onSubmit">确定</el-button>
            <el-button @click="onCancel">取消</el-button>
          </el-form-item>
        </el-form>
      </el-col>
    </el-row>
    <el-dialog title="菜品预览" v-model="dialogVisible">
      <img w-full :src="dialogImageUrl" alt="Preview Image" />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

import request, { baseURL } from '../utils/request'
import api from '../utils/api'

const form = reactive({
  id: '',
  name: '',
  price: '',
  fileList: []
})
const route = useRoute()
Object.assign(form, { ...route.query })
onMounted(() => {
  queryPics(form.id)
})
// 根据 id 查询当前菜品图片
const queryPics = async (id) => {
  try {
    const res = await request({
      url: api.QUERY_PICS,
      method: 'GET',
      params: {
        dish_id: +id
      }
    })
    const fileList = res.pics.map((pic) => {
      const { pic_id, img_path_web } = pic
      return {
        url: `${baseURL}${img_path_web}`,
        name: pic_id
      }
    })
    Object.assign(form, { ...route.query, fileList })
  } catch (e) {
    console.log('e', e)
    ElMessage.error('服务器异常,请稍后再试')
  }
}
//
const addPicsCache = reactive([])
const pictureChangeHandler = (file, fileList) => {
  form.fileList = fileList
  addPicsCache.push(file)
}
// 预览
const dialogImageUrl = ref('')
const dialogVisible = ref(false)
const picturePreviewHandler = (file, fileList) => {
  dialogImageUrl.value = file.url
  dialogVisible.value = true
}
// 删除图片
const removePicsCache = reactive([])
const pictureRemoveHandler = (file, fileList) => {
  form.fileList = fileList
  const reg = /^[0-9]+$/
  if (reg.test(file.name)) {
    removePicsCache.push(file.name)
  } else {
    const index = addPicsCache.findIndex((pic) => file.name === pic.name)
    addPicsCache.splice(index, 1)
  }
}
const removePics = async (ids) => {
  await request({
    url: api.DELETE_PIC,
    method: 'POST',
    data: {
      pic_ids: ids
    }
  })
}
// 校验 form
const verifyForm = (name, price) => {
  if (!name) {
    ElMessage({
      type: 'warning',
      message: '请填写菜品名称'
    })
    return false
  }
  if (!price) {
    ElMessage({
      type: 'warning',
      message: '请填写菜品价格'
    })
    return false
  }
  return true
}
// 修改菜品名称-价格
const dishEditHandler = async (id, name, price) => {
  await request({
    url: api.EDIT_DISH,
    method: 'POST',
    data: {
      dish_id: +id,
      dish_name: name,
      dish_price: price
    }
  })
}
// 上传图片
const uploadPics = async (id, file) => {
  const data = new FormData()
  data.append('dish_id', id)
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
// 确认
const onSubmit = async () => {
  const { id, name, price } = form
  if (verifyForm(name, price)) {
    try {
      await dishEditHandler(id, name, price)
      await removePics(removePicsCache)
      const uploadPicsList = addPicsCache.map((file) => uploadPics(id, file))
      await Promise.all(uploadPicsList)
      ElMessage.success('编辑成功')
      setTimeout(() => router.push('/'), 1000)
    } catch (e) {
      ElMessage.error('菜品上传错误,请稍后重试')
      console.log('e', e)
    }
  }
}
// 取消
const router = useRouter()
const onCancel = () => router.push('/')
</script>

<style scoped>
.form-box {
  width: 60%;
}
</style>