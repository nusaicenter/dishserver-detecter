<template>
    <div>
        <div v-if="images.length==0" class="noImage">暂无图片</div>

        <div class="images" @keydown="handleKeys" @keyup="handleKeysUp">
            <div v-for="(item, id) in partOfImage" class="image-middle">
                <div style="position: relative">
                    <el-image class="image" :src="item.path" :title="item.key" :preview-src-list="partOfImage.map((e)=>e.path)" :initial-index="id" :hide-on-click-modal="true" />
                    <!-- <el-image class="image" :src="item.path" /> -->
                    <el-checkbox v-model="item.select" size="large" class="checkbox" @change="checkboxChange(id)" />
                </div>
                <text v-if="item.class">{{item.class}}</text>
            </div>
        </div>
        <div v-if="images.length>0" style="display: flex; justify-content: center;">
            <el-pagination layout="prev, pager, next" :current-page="pagesInfo.curPage" @update:current-page="changePage"
            :page-size="pagesInfo.size"  :total="images.length" />
        </div>
        <div v-if="images.length>0">
            <el-button type="default" @click="selectAll">全选</el-button>
            <el-button type="default" @click="selectToggle">反选</el-button>
            <slot />
        </div>
    </div>
</template>

<script setup>
import { onMounted, reactive, ref, watch } from 'vue';
// images = [{
//     path: string,
//     key: string,
//     select: bool
// }]
const props = defineProps(['images'])
const emits = defineEmits(['delete'])

const isShift = ref(false)
const prevSelectId = ref(0)
const prevPage = ref(0)
const partOfImage = ref([])
const pagesInfo = reactive({
    curPage: 1,
    size: 20,
})
onMounted(()=>{
    changePage(1)
})

watch(()=>props.images,(newVal, oldVal)=>{
    // refresh partOfImage cache to new data
    changePage(prevPage.value == 0 ? 1 : prevPage.value)
})


const selectAll = () => {
    // props.images.slice(startIndex, endIndex).forEach(e => {
    partOfImage.value.forEach(e => {
        e.select = true
    });
}
const selectToggle = () => {
    partOfImage.value.forEach(e => {
        e.select = !e.select
    });
}

const handleKeys = (e) => {
    switch (e.key.toLowerCase()) {
        case 'd':
            prevPage.value = pagesInfo.curPage
            emits('delete')
            break
        case 'arrowleft':
        case 'a':
            changePage(pagesInfo.curPage - 1)
            break
        case 'arrowright':
        case 's':
            changePage(pagesInfo.curPage + 1)
            break
        case 'shift':
            isShift.value = true
            break
    }
}

const handleKeysUp = (e) => {
    if (e.key == 'Shift') {
        isShift.value = false
        console.log("shift", isShift.value);
    }
}

const checkboxChange = (id) => {
    if (isShift.value == true) {
        // select all images between current and previous selected
        let startId = 0
        let endId = 0
        if (id < prevSelectId.value) {
            startId = id
            endId = prevSelectId.value
        } else {
            startId = prevSelectId.value
            endId = id
        }
            let newVal = partOfImage.value[prevSelectId.value].select
        for (let i = startId; i <= endId; i++) {
            partOfImage.value[i].select = newVal
        }
    } else {
        prevSelectId.value = id
    }
}

const changePage = (newPage) => {
    // debugger
    let maxPageID = Math.ceil(props.images.length/pagesInfo.size)
    pagesInfo.curPage = Math.max(1, Math.min(newPage, maxPageID))
    prevPage.value = 0
    let page = pagesInfo.curPage - 1
    let startIndex = page * pagesInfo.size
    let endIndex = Math.min((page + 1) * pagesInfo.size, props.images.length)
    partOfImage.value = props.images.slice(startIndex, endIndex)
}

</script>

<style scoped>

.image {
  height: 120px;
}

.images {
  display: flex;
  margin-top: 5px;
  margin-left: 5px;
  margin-right: 5px;
  flex-wrap: wrap;
}

.image-middle {
  display: flex;
  flex-direction: column;
  align-items: center;
  /* position: relative; */
  margin-right: 5px;
  margin-bottom: 5px;
}
.checkbox {
  position: absolute;
  right: 6px;
  bottom: -1px;
}
.noImage {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 80px;
    color: var(--el-text-color-regular)
}
</style>
