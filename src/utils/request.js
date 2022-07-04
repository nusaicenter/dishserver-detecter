import axios from 'axios'

const baseURL = import.meta.env.PROD ? '/backend' : 'http://192.168.100.103:30021'
// const baseURL = 'http://192.168.100.103:30021'

const service = axios.create({
  baseURL,
  timeout: 5000
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
    const { data: { data, error_code } } = response
    return error_code === 0 ? Promise.resolve(data) : Promise.reject(response)

  },
  error => {
    console.log('service response error', error)
    Promise.reject(error)
  })

export { baseURL }
export default service