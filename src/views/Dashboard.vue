<template>
  <div class="container">
    <div class="handle-box">
      <el-input v-model="search" placeholder="菜名" class="handle-input mr10"></el-input>
      <el-button type="primary" @click="handleSearch">
        <el-icon>
          <Search />
        </el-icon>搜索
      </el-button>
    </div>
    <el-table
      :data="dish.list"
      border
      class="table"
      ref="multipleTable"
      header-cell-class-name="table-header"
    >
      <el-table-column type="index" label="序号" width="100" align="center"></el-table-column>
      <el-table-column prop="dish_name" label="菜品名称" align="center"></el-table-column>
      <el-table-column label="菜品价格" width="200" align="center">
        <template #default="scope">￥{{ scope.row.dish_price }}</template>
      </el-table-column>
      <el-table-column label="菜品预览" align="center">
        <template #default="scope">
          <el-image
            class="table-td-thumb"
            :src="scope.row"
            :preview-src-list="[scope.row.dish_src]"
          >
            <template #error>
              <div class="image-slot">
                <el-icon><Picture /></el-icon>
              </div>
            </template>
          </el-image>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" align="center">
        <template #default="scope">
          <el-button type="text" icon="el-icon-edit" @click="handleEdit(scope.row)">编辑</el-button>
          <el-button
            type="text"
            icon="el-icon-delete"
            class="red"
            @click="handleDelete(scope.row.dish_id)"
          >删除</el-button>
        </template>
      </el-table-column>
    </el-table>
    <div class="pagination">
      <el-pagination
        background
        layout="prev, pager, next"
        :total="total"
        :current-page="current"
        @current-change="handlePageChange"
      ></el-pagination>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'

import api from '../utils/api'
import request from '../utils/request'

const dish = reactive({ list: [] })

const total = ref(0) // 菜品总个数
// 请求所以已上传的菜品-图片
const requestDishes = async (current = 1) => {
  try {
    const list = await request({
      url: api.QUERY_DISH,
      method: 'GET',
      page: current // TODO: 分页
    })
    dish.list = list
    total.value = dish.list.length || 50
  } catch (e) {
    console.log('e', e)
    ElMessage({
      type: 'error',
      message: '服务器异常,请稍后再试'
    })
  }
}

onMounted(() => {
  requestDishes()
})

// 编辑操作
const router = useRouter()
const handleEdit = ({ dish_id, dish_name, dish_price }) => {
  router.push({
    path: '/edit',
    query: { id: dish_id, name: dish_name, price: dish_price }
  })
}
// 删除操作
const handleDelete = (id) => {
  ElMessageBox.confirm('确定要删除吗？', '提示', {
    type: 'warning',
    confirmButtonText: '确认',
    cancelButtonText: '取消'
  })
    .then(() => dishDelete(id))
    .catch(() => {
      ElMessage({
        type: 'info',
        message: '取消删除'
      })
    })
}
// 菜品接口删除
const dishDelete = async (id) => {
  try {
    await request({
      url: api.DELETE_DISH,
      method: 'POST',
      data: {
        dish_id: id
      }
    })
    const index = dish.list.findIndex((dish) => dish.dish_id === id)
    dish.list.splice(index, 1)
    ElMessage.success('删除成功')
  } catch (e) {
    ElMessage({
      type: 'error',
      message: '删除失败'
    })
  }
}
// TODO:查询
const search = ref('')
const handleSearch = () => {
  console.log('search', search)
}
// TODO:分页查询
const current = ref(1) // 默认第 1 页
const handlePageChange = (val) => {
  console.log('val', val)
  // requestDishes(2)
}
</script>

<style scoped>
.handle-box {
  margin-bottom: 20px;
  align-self: left;
}

.handle-select {
  width: 120px;
}

.handle-input {
  width: 300px;
  display: inline-block;
}
.table {
  width: 100%;
  font-size: 14px;
}
.red {
  color: #ff0000;
}
.mr10 {
  margin-right: 10px;
}
.table-td-thumb {
  display: block;
  margin: auto;
  width: 40px;
  height: 40px;
}
.image-slot{
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: var(--el-fill-color-light);
  color: var(--el-text-color-secondary);
  font-size: 30px;
}
</style>
