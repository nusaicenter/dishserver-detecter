import axios from 'axios'

const baseURL = 'http://127.0.0.1:36192'

const service = axios.create({
  baseURL,
  timeout: 1000*60*10
})

service.interceptors.request.use(
  config => {
    return config
  },
  error => {
    console.log('service request error', error)
    Promise.reject(error)
  })

service.interceptors.response.use(
  response => {
    return Promise.resolve(response) 
    // const { data: { data, error_code } } = response
    // return error_code === 0 ? Promise.resolve(data) : Promise.reject(response)
  },
  error => {
    console.log('service response error', error)
    return Promise.resolve(error.response)
  })

export { baseURL }
export default service