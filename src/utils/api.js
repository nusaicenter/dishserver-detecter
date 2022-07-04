const api = {
  QUERY_DISH: '/dish/', // 获取所有菜品
  QUERY_PICS: '/dish/pics', // 根据 id 获取对应菜品图片 dish_id: number
  ADD_DISH: '/dish/add', // 添加菜品 dish_name: string | dish_price: string
  ADD_PIC: '/dish/pic/add', // 添加菜品图片 dish_id: number | img: File
  DELETE_PIC: '/dish/pic/del', // 删除菜品图片 pic_ids: array<number>
  DELETE_DISH: '/dish/del', // 删除菜品图片 dish_id: number
  EDIT_DISH: '/dish/edit' // 编辑菜品 dish_id: number [| dish_name: string | dish_price: string]
}
export default api
