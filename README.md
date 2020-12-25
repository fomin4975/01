!!!
ubuntu + windows 10 dual boot
https://help.ubuntu.ru/wiki/%D0%BF%D0%B5%D1%80%D0%B5%D1%85%D0%BE%D0%B4%D0%B8%D0%BC_%D0%BD%D0%B0_ubuntu-linux_-_%D0%B8%D0%BD%D1%81%D1%82p%D1%83%D0%BA%D1%86%D0%B8%D1%8F_%D0%B4%D0%BB%D1%8F_%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8F_windows
https://linuxconfig.org/how-to-install-ubuntu-20-04-alongside-windows-10-dual-boot
https://itsfoss.com/install-ubuntu-1404-dual-boot-mode-windows-8-81-uefi/

http://releases.ubuntu.com/


https://medium.com/analytics-vidhya/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a
https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0


How to install the NVIDIA drivers on Ubuntu 20.04 Focal Fossa Linux
https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux

How to install CUDA on Ubuntu 20.04 Focal Fossa Linux
https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux


!
https://habr.com/ru/post/520996/#a96

!
https://linustechtips.com/topic/1261536-linking-two-rtx-3090-but-from-different-brands-is-this-doable/?tab=comments#comment-14149037
https://www.ixbt.com/3dv/palit-geforce-rtx-3090-gamerock-oc-review.html





https://blog.exxactcorp.com/whats-the-best-gpu-for-deep-learning-rtx-2080-ti-vs-titan-rtx-vs-rtx-8000-vs-rtx-6000/
https://www.pugetsystems.com/labs/hpc/2-x-RTX2070-Super-with-NVLINK-TensorFlow-Performance-Comparison-1551/?__cf_chl_captcha_tk__=ca08ddfccf84eafe30e8fb6d369c6bcbbf49638e-1608816023-0-AWks9_ZLQ6VSbzGRBrKV7HI_L_dZIpQXZLrIT2KeTgrbvol0hRv3cdhUkVG-Upkg20uz_wIYPiErx82212ET266MEeBjayV1f0lHfYY8F7zby2tbjGiRA-rf_uUiA9xGw7d1QIVOQKoW2YRzyOQi_7F6qJLLHFOhwPoZARXmERs86O6AGLQt7zjByavUaJsUHUuoplb3GEOj87MgYEfZ5t-Lb-WQCLJu20PkQHMxh8UfEqGdPkINtrLm6b9lTb64QlBLJwr-GeomU5PeUub3ef05H_WL-YENFPunh0HPbFRkQIvLKYxAeeoWlFkWpNejyXNj6KsHRel9QLX1bC_nrsLNiGzUhg07ZMWIzDdm0kdW0S-CI9YSfFCMLuUamJW02D0dPDyoVTwD-5LSQeg8Y2oBwhcsPeT0Y_RU4q3uTVXBg0hku-Gj2bV-11tVAdHrw4G5jgsleWVeCf3ug27dBF_o60apyvIRLKdjTibVNl1xBFTKaXBhOhGlgs9oEJIHbPEZZGnY0fLVsiDj46pzaBFWDRwMKU1ysJFsnu9AfbUUNHr_szL3QVnZCeG-BzfsbTF7-bLRt5Vvt7eq4cr1FwzJa9DiD9dTj45155zuRvzeHJllQy7N6wDj1_pC2vyUiw
https://www.reddit.com/r/MachineLearning/comments/ily622/d_nvlink_release_for_rtx_3090/
https://www.pugetsystems.com/labs/hpc/RTX3090-TensorFlow-NAMD-and-HPCG-Performance-on-Linux-Preliminary-1902/
https://www.reddit.com/r/nvidia/comments/iovn4f/the_3090_is_not_a_titan_replacement_and_here_is/
https://www.techpowerup.com/272525/nvidia-rtx-3090-performance-10-15-higher-than-rtx-3080-in-4k
https://bizon-tech.com/blog/best-gpu-for-deep-learning-rtx-2080-ti-vs-titan-rtx-vs-rtx-8000-vs-rtx-6000
https://linuxhint.com/best_graphics_cards_deep_learning/
--------------
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"],
                      cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
                         all_reduce_alg="hierarchical_copy")
                                                   )
-------------
https://www.pugetsystems.com/labs/hpc/RTX-2080Ti-with-NVLINK---TensorFlow-Performance-Includes-Comparison-with-GTX-1080Ti-RTX-2070-2080-2080Ti-and-Titan-V-1267/?__cf_chl_captcha_tk__=34f015be718c14a4288c9e4b36a2ff9713d3a555-1608885400-0-AZuKgAZVnPpFXRQXr6_mbSXpDpJN5qu_6QnqnYgCSnO9lE74BMQDrPLySMZHulD6JdrzqQ7UDNBva_XSUHxQeCcOkFtd6kJbzkH1YhfNBdIUrA1EEANPs00ZL6cbQszlKD3irYs01g27eJc_DzEysveEaUdexOZ_yphh-u_dtvCCyH1besJdATCFTBiYyJ9qV0lhM7NS3VX_MZ7AaQeNVViO_Hpqo7OqS0Bm4o2bvgcT0t3J5mburq5O61-WW3qpUALgvEVOLkOrqR9N5TuFFgiuqq-PiXc4-FYf99Id8uCCgHGGNXCAg4JLjtgPquVigLAB2cgvKjCUz6QTd0ioSC0O8qGwiTaN2RVZT2uOASoqDRgi7790FfkLNymXeJrWXvXGyiGozgQR8Fmq0Grqyf6c_Aa9eYXl0GRfXQRlOMLwcab7CRdCI-krIzHQ8K26U1TK7cJnESoqC0K4oKkvYHEGbRxRPZSn00uxzRY1zrg8Lnhjn0AXpwc2M_hz0H6NmXilXmIB7_MmpsTN_0cpe7xeNaPkUYHoDiCwSHmmw1DBuhsQBuQu0__QJph9oK2-pE-watlYPCkfPGxiCqrXUj_nxO3q3PXIBKwyfuPF1aEkl2VyBbGZOlVBzxkwCEK0mlwzleqX9bP0nGUmvU2u2unvtfQNkF5TzpEqvHxa1MoY1k8nA7-KPj3VtTdUV0WUSQ
https://developer.nvidia.com/explicit-multi-gpu-programming-directx-12?ncid=afm-chs-44270&ranMID=44270&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-fL83Afnqn7XklRisqvdQVA
https://forums.fast.ai/search?q=nvlink%20topic%3A78202
https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/
https://keras.io/guides/distributed_training/
https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy
https://www.tensorflow.org/guide/distributed_training
https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
https://datascience.stackexchange.com/questions/46952/should-i-connect-my-two-gpus-with-sli-or-not-for-keras-tensorflow
