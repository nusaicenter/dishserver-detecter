<template>

  <div>
    <div class="container">
      <div style="display: flex; align-items: center;">

        <!-- <el-button type="primary" @click="refreshDir" style="margin:10px">展示处理结果</el-button> -->
        <label class="el-title">项目 </label>
        <el-select size="default" v-model="data.dir_path" placeholder="请选择菜品图片文件夹" style="margin:10px; width: 300px;">
          <el-option v-for="path in data.allImageFolder" :label="getNameFromPath(path)" :value="path"
            @click="refreshDir" />
        </el-select>
        <el-button type="primary" @click="refreshDirlist" style="margin: 0px 10px">
          <el-icon>
            <RefreshRight />
          </el-icon>
        </el-button>
        <el-switch @change="switchViewType" v-model="data.viewConfirm" size="large"
          :active-text="'已确认图片 '+confirmObjs.length" :inactive-text="'待确认图片 '+subimgObjs.length" />
      </div>

      <div style="margin-top: 15px; display: flex; align-items: center; ">
        <label class="el-title">按类别名称筛选</label>
        <!-- <el-checkbox-group v-model="data.checkboxFilter" > -->
        <el-radio-group v-model="data.checkboxFilter" >
          <!-- <el-checkbox-button v-for="cls in data.clsList" :label="cls">{{cls}}</el-checkbox-button> -->
          <el-radio-button >全部</el-radio-button>
          <el-radio-button v-for="cls in data.clsList" :label="cls">{{cls}}</el-radio-button>
        </el-radio-group>
      </div>
      <el-divider />

      <div>
        <label class="el-title">待确认菜品图片</label>
        <image-list :images="imageObjs">

          <el-button type="danger" @click="deleteSubImg">删除菜品</el-button>
          <el-button type="primary" @click="confirmSubimg">确认已选图片类别</el-button>
          <el-button type="primary" @click="autoLabelSubimg">自动标注图片</el-button>
          <div style=" margin:20px 0px;">
          <label class="el-title">修改类别</label>
            <el-select v-model="data.clsName" placeholder="修改类别" style="margin-right:15px">
              <el-option @click="addClsName" value="修改类别">+ 添加类别</el-option>
              <el-option v-for="name in data.clsList" :value="name"></el-option>
            </el-select>
            <el-button type="primary" @click="setClsName">确认修改</el-button>
          </div>
        </image-list>

      </div>

    </div>

  </div>
</template>

<script setup>
import { reactive, onMounted, watch, ref } from 'vue'
import request from '../utils/request'
import { ElLoading, ElMessage, ElMessageBox } from 'element-plus'
import ImageList from './components/ImageList.vue'
import {getImageObjKeys, getNameFromPath} from './components/utils' 


const data = reactive({
  allImageFolder: [],
  allSubImagePath: [],
  // confirmedSubImage: [],
  selectedSubImage: [],
  checkboxFilter: '',//选择用来筛选子图的类名
  clsList: [],
  pageNum: 1,
  isShift: false,
  viewConfirm: false,
  curPage: 1, //当前页面序号
  dir_path: '',
  clsName: '', //要修改为的类名
  newClsName: '', //要添加的新类名
})


const subimgObjs = ref([])
const confirmObjs = ref([])
const imageObjs = ref([])


watch(
  ()=> data.checkboxFilter, (newValue, oldValue)=>{
    refreshDir()
  }
)
// watch(
//   ()=> data.viewConfirm, (newValue, oldValue)=>{
//     console.log("newValue", newValue);

//   }
// )

onMounted(() => {
  refreshDirlist()
  data.clsName = ''
})



const refreshDirlist = async () => {
  const res = await request({
    url: '/get_directories',
    method: 'GET',
  })
  console.log("get dir list ok", res.data);
  data.allImageFolder = res.data
  data.dir_path = ''
}
const refreshDir = async () => {
  getSubimg()
}

const getSubimg = async () => {
  const res = await request({
    url: '/get_subimg',
    method: 'POST',
    data: {
      path: data.dir_path,
      labels: data.checkboxFilter,
    }
  })
  data.allSubImagePath = res.data.subimg
  data.selectedSubImage = Array(data.allSubImagePath.length).fill(false)
  // data.confirmedSubImage = res.data.confirmed

  subimgObjs.value = []
  confirmObjs.value = []
  for (let i=0;i<res.data.subimg.length;i++){
    let img = res.data.subimg[i]
    subimgObjs.value.push({
      path: img.path,
      key: img.key,
      class: img.class,
      select: false
    })
  }

  for (let i=0;i<res.data.confirmed.length;i++){
    let img = res.data.confirmed[i]
    confirmObjs.value.push({
      path: img.path,
      key: img.key,
      class: img.class,
      select: false
    })
  }
  getClsList()
  switchViewType()
}

const switchViewType =(e) =>{
    if (data.viewConfirm){
      imageObjs.value = confirmObjs.value
    }else{
      imageObjs.value = subimgObjs.value
    }
}

const getClsList = async () => {
  const res = await request({
    url: '/get_cls_list',
    method: 'GET',
  })
  data.clsList = res.data
}



const deleteSubImg = async()=>{

  let keys = getImageObjKeys(data.viewConfirm ? confirmObjs : subimgObjs)
  const res = await request({
    url: '/delet_subimgs',
    method: 'POST',
    data: {
      key: keys
    }
  })
  ElMessage({ message: '删除成功', type: 'success' })
  refreshDir()


}

const setClsName =async () =>{
  let keys = getImageObjKeys(data.viewConfirm ? confirmObjs : subimgObjs)

  const res = await request({
    url:'/set_cls_name',
    method: 'POST',
    data:{
      key: keys,
      cls_name: data.clsName,
    }
  })
  ElMessage({ message: '修改成功', type: 'success' })
  refreshDir()


}

const addClsName = () => {

  ElMessageBox.prompt('', '新增类别名', {
    confirmButtonText: '确认',
    cancelButtonText: '取消',
  })
    .then(({ value }) => {
      request({
        url: '/add_cls_name',
        params: {
          cls_name: value,
        }
      }).then(
        (res) => {
          getClsList()
          console.log("value", value);
          data.clsName = value
          ElMessage({ message: '添加成功', type: 'success' })
        }
      ).catch((err) => {
        ElMessage({ message: '添加失败', type: 'error' })
      })
    }).catch((err)=>{
      console.log("cancel add new class" );
    })
}


const confirmSubimg = async () =>{
  let keys = getImageObjKeys(data.viewConfirm ? confirmObjs : subimgObjs)


  const res = await request({
    url: '/confirm_subimgs',
    method: 'POST',
    data: {
      key: keys
    }
  })
  console.log("res.data", res.data);
  ElMessage({ message: '已确认', type: 'success' })
  refreshDir()
  
}

const autoLabelSubimg = async () => {
  const loadingInstance = ElLoading.service({
    text: '正在自动标注',
    background: 'rgba(0, 0, 0, 0.6)',
  })
  const res = await request({
    url: '/auto_label_subimg',
    method: 'GET',
  })
  loadingInstance.close()

  ElMessage({ message: res.data, type: (res.status == 200) ? 'success' : 'warning' })
  refreshDir()
}


</script>

<style scoped>
.container {
  margin-bottom: 10px;
}

.el-title{
  color: var(--el-text-color-regular);
  margin-right: 15px;
  font-size: 15px;
}
</style>