<template>

  <div>
    <div class="container">
      <div style="display: flex; align-items: center;">

        <!-- <label class="el-title">   </label> -->
        <el-select size="default" v-model="data.dir_path" placeholder="请选择餐盘图片文件夹" style="margin:10px; width: 300px;">
          <el-option v-for="path in data.folderPaths" :label="getNameFromPath(path)" :value="path"
            @click="refreshDir" />
        </el-select>
        <el-button type="primary" @click="refreshDirlist" style="margin: 0px 10px">
          <el-icon>
            <RefreshRight />
          </el-icon>
        </el-button>

        <el-button v-if="data.dir_path" type="primary" @click="data.isUpload = !data.isUpload" style="margin: 0px 10px">
          上传图片
        </el-button>
        <el-button v-if="data.isUpload" type="warning" @click="uploadImage" style="margin: 0px 10px">
          确认上传
        </el-button>

      </div>
      <div v-show="data.isUpload">
        <el-upload action="#" list-type="picture-card" accept="image/*" :limit="10" :multiple="true"
          :auto-upload="false" :file-list="data.uploadFiles" :on-change="pictureChangeHandler"
          :on-remove="pictureRemoveHandler" >
        <el-icon>
          <Plus />
        </el-icon>
        </el-upload>
        <el-divider></el-divider>
      </div>


      <!-- :page-count="data.pageNum"  :current-page="data.curPage"  :page-size="data.pageSize"   @update:current-page="changeMainImgPage"-->
      <image-list :images="imageObjs" @delete="deleteImg">
        <el-button v-if="!data.isSimilar" type="warning" @click="findSimilar">仅看相似图片</el-button>
        <el-button v-if="data.isSimilar" type="warning" @click="findSimilar">查看全部图片</el-button>
        <el-button type="danger" @click="deleteImg">删除已选图片</el-button>
        <el-button type="primary" @click="detectBox(false)">检测已选</el-button>
        <el-button type="primary" @click="detectBox(true)">检测全部</el-button>
        <!-- <el-button type="danger" @click="testBtn">test</el-button> -->
      </image-list>



    </div>

  </div>
</template> 

<script setup>
import { computed } from '@vue/reactivity';
import { ElLoading, ElMessage } from 'element-plus';
import { reactive, onMounted, ref } from 'vue'
import request from '../utils/request'
import ImageList from './components/ImageList.vue'
import { getImageObjKeys, getNameFromPath } from './components/utils'

const data = reactive({
  folderPaths: [],
  allMainimgs: [],
  dir_path: '',
  uploadFiles: [],
  isSimilar: false,
  isUpload: false
})

const imageObjs = computed(() => {
  let objs = []
  for (let i = 0; i < data.allMainimgs.length; i++) {
    objs.push({
      path: data.allMainimgs[i].path,
      key: data.allMainimgs[i].key,
      select: false
    })
  }
  return objs
})

onMounted(() => {
  refreshDirlist()
})


const refreshDirlist = async () => {
  const res = await request({
    url: '/get_directories',
    method: 'GET',
  })
  console.log("get dir list ok", res.data);
  data.folderPaths = res.data
  data.dir_path = ''
  data.isSimilar = false
  data.isUpload = false
  data.allMainimgs = []
}


const refreshDir = async () => {

  const loadingInstance = ElLoading.service({
    text: '正在加载',
    background: 'rgba(0, 0, 0, 0.6)',
  })

  const res = await request({
    url: '/get_mainimg',
    method: 'POST',
    data: {
      path: data.dir_path,
    }
  })
  data.allMainimgs = res.data.mainimg
  data.isSimilar = false
  // data.pageNum = res.data.page_num
  // data.selectedImage = Array(data.allMainimgs.length).fill(false)
  loadingInstance.close()

}


const detectBox = async (all) => {
  let keys = []
  if (all == true) {
    keys = data.allMainimgs.map(e => e.key)
  } else {
    keys = getImageObjKeys(imageObjs)
  }

  if (keys.length > 0) {
    const loadingInstance = ElLoading.service({
      text: '正在检测菜品图片',
      background: 'rgba(0, 0, 0, 0.6)',
    })
    const res = await request({
      url: '/detect_box',
      method: 'POST',
      data: {
        path: keys
      }
    })
    loadingInstance.close()
    refreshDir()
  } else {
    ElMessage({ message: '请选择图片', type: 'warning' })
  }
}

const deleteImg = async () => {
  let keys = getImageObjKeys(imageObjs)
  if (keys.length > 0) {

    const res = await request({
      url: '/delete_images',
      method: 'POST',
      data: {
        path: keys
      }
    })
    // data.selectedImage = []
    refreshDir()
  } else {
    ElMessage({ message: '请选择图片', type: 'warning' })
  }
}

const testBtn = () => {

}

const findSimilar = async () => {
  if (data.isSimilar) {
    // if in similar mode, switch back to view all images
    data.isSimilar = false
    refreshDir()
  } else {
    // if in normal mode, switch to similar mode
    let keys = getImageObjKeys(imageObjs)
    if (keys.length > 0) {
      const res = await request({
        url: '/find_similar',
        method: 'POST',
        data: {
          keys: keys
        }
      })
      data.allMainimgs = res.data
      data.isSimilar = true
    } else {
      ElMessage({ message: '请选择图片', type: 'warning' })
    }
  }
}

const uploadImage = async () => {
  if (data.uploadFiles.length == 0){
    ElMessage({ message: '请添加图片', type: 'warning' })
    return
  }
  const formdata = new FormData()
  data.uploadFiles.forEach(el => {
    formdata.append('img', el.raw)
  });

  const res = await request({
    url: '/mainimg/add_mainimg',
    method:'POST',
    headers: {'Content-Type': 'multipart/form-data'},
    data: formdata
  })
  ElMessage({ message: '上传成功', type: 'success' })
  // 清空上传列表
  data.uploadFiles = []
  data.isUpload = false
  refreshDir()
}

const pictureChangeHandler = (file, fileList) => {
  data.uploadFiles = fileList
}
const pictureRemoveHandler = (file, fileList) => {
  data.uploadFiles = fileList
}

</script>

<style scoped>
.container {
  margin-bottom: 10px;
}
</style>