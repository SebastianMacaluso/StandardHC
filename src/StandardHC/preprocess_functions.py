#!/usr/bin/env python

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import copy

#-------------------------------------------------------------------------------------------------------------
#/////////////////////    AUXILIARY FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
# We want to center the image so that the total pT weighted centroid pixel is at (eta,phi)=(0,0). So we calculate eta_center,phi_center
def center(tracks,towers):
# format for subjets: [[pT1,pT2,...],[eta1,eta2,....],[phi1,phi2,....]]

#  print('Calculating the image center for the total pT weighted centroid pixel is at (eta,phi)=(0,0) ...')
#  print('-----------'*10)

#  Nsubjets=len(Subjets[0])
#  print len(tracks),len(towers)
#  print len(tracks[0]),len(towers[0])
 
  pTj=np.sum(tracks[0])+np.sum(towers[0])

  eta_c=(np.sum(tracks[0]*tracks[1])+np.sum(towers[0]*towers[1]))/pTj 
  phi_c=(np.sum(tracks[0]*tracks[2])+np.sum(towers[0]*towers[2]))/pTj 

  return eta_c,phi_c


##---------------------------------------------------------------------------------------------
# Shift the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates
def shift(subjets,Eta_c,Phi_c):
#  print('Shifting the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates ...')
#  print('-----------'*10)

  subjets[1]=subjets[1]-Eta_c
  subjets[2]=subjets[2]-Phi_c

  return subjets
  
  
##---------------------------------------------------------------------------------------------
# Calculate DeltaR for each subjet in the shifted coordinates and the angle theta of the principal axis
def principal_axis(tracks,towers):
#  print('Getting DeltaR for each subjet in the shifted coordinates and the angle theta of the principal axis ...')
#  print('-----------'*10)

  tan_theta=0.
  M11=np.sum(tracks[0]*tracks[1]*tracks[2])+np.sum(towers[0]*towers[1]*towers[2])
  M20=np.sum(tracks[0]*tracks[1]*tracks[1])+np.sum(towers[0]*towers[1]*towers[1])
  M02=np.sum(tracks[0]*tracks[2]*tracks[2])+np.sum(towers[0]*towers[2]*towers[2])  
  denom=(M20-M02+np.sqrt(4*M11*M11+(M20-M02)*(M20-M02)))
  if(denom!=0):
    tan_theta=2*M11/denom


  return tan_theta


##---------------------------------------------------------------------------------------------
# Rotate the coordinate system so that the principal axis is the same direction (+ eta) for all jets
def rotate(subjets,tan_theta):
#  print('Rotating the coordinate system so that the principal axis is the same direction (+ eta) for all jets ...')
#  print('-----------'*10)

  rotpt=subjets[0]
  roteta=subjets[1]*np.cos(np.arctan(tan_theta))+subjets[2]*np.sin(np.arctan(tan_theta))
  rotphi=np.unwrap(-subjets[1]*np.sin(np.arctan(tan_theta))+subjets[2]*np.cos(np.arctan(tan_theta)))

  return [rotpt,roteta,rotphi,subjets[3],subjets[4]]


##---------------------------------------------------------------------------------------------
# BEFORE PIXELATING. Reflect the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane
def ver_flip_const(tracks,towers): 
  
#  print('Flipping the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane ...')
#  print('-----------'*10)

#   print('towers=',towers)
#   print('---'*20)
#   print('transpose towers=',np.transpose(towers))
#   print('---'*20)
#   print('---'*20)
#   print('tracks=',tracks)
#   print('---'*20)
#   print('transpose tracks=',np.transpose(tracks))
#   print('---'*20)
#   sys.exit()

  left_tracks=[track[0] for track in tracks if track[1]<0]
  left_towers=[tower[0] for tower in towers if tower[1]<0]

  right_tracks=[track[0] for track in tracks if track[1]>0]
  right_towers=[tower[0] for tower in towers if tower[1]>0]

  left_sum=np.sum(left_tracks)+np.sum(left_towers)
  right_sum=np.sum(right_tracks)+np.sum(right_towers)   

  if left_sum>right_sum:
      flip_tracks = tracks
      flip_towers = towers 
  else:
      flip_tracks = [[track[0],-track[1],track[2],track[3],track[4]] for track in tracks]
      flip_towers = [[tower[0],-tower[1],tower[2]] for tower in towers]
   
#   print('-----------'*10)    
#   print('Left sum=',left_sum) 
#   print('Right sum=',right_sum)     
#   print('-----------'*10)
#   print('Track before ver flip=',tracks[0])
#   print('Track after ver flip=',flip_tracks[0])
#   print('-----------'*10)
#   print('Tower before ver flip=',towers[0])
#   print('Tower after ver flip=',flip_towers[0]) 
#   print('-----------'*10)
#   print('-----------'*10)
  
  return flip_tracks,flip_towers   
  

##---------------------------------------------------------------------------------------------
#  BEFORE PIXELATING. Reflect the image with respect to the horizontal axis to ensure the 3rd maximum is on the upper half-plane
def hor_flip_const(tracks,towers): 
  
#  print('Flipping the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane ...')
#  print('-----------'*10)

  lower_tracks=[track[0] for track in tracks if track[2]<0]
  lower_towers=[tower[0] for tower in towers if tower[2]<0]

  upper_tracks=[track[0] for track in tracks if track[2]>0]
  upper_towers=[tower[0] for tower in towers if tower[2]>0]

  lower_sum=np.sum(lower_tracks)+np.sum(lower_towers)
  upper_sum=np.sum(upper_tracks)+np.sum(upper_towers)   

  if lower_sum<upper_sum:
      flip_tracks = tracks
      flip_towers = towers 
  else:
      flip_tracks = [[track[0],track[1],-track[2],track[3],track[4]] for track in tracks]
      flip_towers = [[tower[0],tower[1],-tower[2]] for tower in towers]
   
#   print('-----------'*10)    
#   print('lower_sum sum=',lower_sum) 
#   print('upper_sum sum=',upper_sum)    
#   print('-----------'*10)
#   print('Track before hor flip=',tracks[0:2])
#   print('Track after hor flip=',flip_tracks[0:2])
#   print('-----------'*10)
#   print('Tower before hor flip=',towers[0:2])
#   print('Tower after hor flip=',flip_towers[0:2]) 
#   print('-----------'*10)
#   print('-----------'*10)
  
  return flip_tracks,flip_towers 
  
  
##---------------------------------------------------------------------------------------------
# Scale the pixel intensities such that sum_{i,j} I_{i,j}=1
def normalize(tracks,towers):
#  print('Scaling the pixel intensities such that sum_{i,j} I_{i,j}=1 ...')
#  print('-----------'*10)
  if len(tracks)>0 and len(towers)>0:
    pTj=np.sum(tracks[0])+np.sum(towers[0])
    tracks[0]=tracks[0]/pTj
    towers[0]=towers[0]/pTj
    
  elif len(tracks)==0:
    pTj=np.sum(towers[0])
    towers[0]=towers[0]/pTj
  
  else:
    pTj=np.sum(tracks[0])
    tracks[0]=tracks[0]/pTj



  return tracks,towers


##---------------------------------------------------------------------------------------------
# Create a coarse grid for the array of pT for the jet constituents, where each entry represents a pixel. We add all the jet constituents that fall within the same pixel 
def create_color_image(tracks,towers,DReta,DRphi,npoints):
    
  ncolors=5

  etamin, etamax = -DReta, DReta # Eta range for the image
  phimin, phimax = -DRphi, DRphi # Phi range for the image

  allimages=[]
  grid=np.zeros((npoints-1,npoints-1,ncolors))
  nonzerogrid=np.zeros((npoints-1,npoints-1))
    
  # Get the position of the track/tower jet constituent in the image. The first pixel is pixel 0, so we shift all the constituents to 0. The divide eta (phi) by the size of a pixel=(length of image/# pixels)  
  if len(tracks)>0:
    ietalisttrack=((tracks[1]+DReta)/(2*DReta/float(npoints-1))).astype(int)
    iphilisttrack=((tracks[2]+DRphi)/(2*DRphi/float(npoints-1))).astype(int)
  
  if len(towers)>0:
    ietalisttower=((towers[1]+DReta)/(2*DReta/float(npoints-1))).astype(int)
    iphilisttower=((towers[2]+DRphi)/(2*DRphi/float(npoints-1))).astype(int)

  ##----------------------------------------
  if len(tracks)>0:
    for ipos in range(len(tracks[0])):
  #     norm=1/float(len(tracks[0]))
       norm=1
       if(0<=ietalisttrack[ipos]<npoints-1 and 0<=iphilisttrack[ipos]<npoints-1): # We ask the constituent to be within the image size
  #       if(tracks[4][ipos]<=2):
  #          ipadd=[0,0,0,0,0,0]
  #       elif(2<tracks[4][ipos]<=4):
  #          ipadd=[norm,0,0,0,0,0]
  #       elif(4<tracks[4][ipos]<=6):
  #          ipadd=[0,norm,0,0,0,0]
  #       elif(6<tracks[4][ipos]<=8):
  #          ipadd=[0,0,norm,0,0,0]
  #       elif(8<tracks[4][ipos]<=10):
  #          ipadd=[0,0,0,norm,0,0]
  #       elif(10<tracks[4][ipos]<=12):
  #          ipadd=[0,0,0,0,norm,0]
  #       else:
  #          ipadd=[0,0,0,0,0,norm]
         toadd=[tracks[0][ipos],0,1,tracks[3][ipos],tracks[4][ipos]] #So images have [track pT,tower pT,track multiplicity,muon multiplicity,track charge] 
  #       toadd.extend(ipadd)
         grid[ietalisttrack[ipos],iphilisttrack[ipos]]=grid[ietalisttrack[ipos],iphilisttrack[ipos]]+toadd

  #       grid[ietalisttrack[ipos],iphilisttrack[ipos]][4]=np.max([grid[ietalisttrack[ipos],iphilisttrack[ipos]][4],])
         nonzerogrid[ietalisttrack[ipos],iphilisttrack[ipos]]=1. #We keep track of the nonzero entries
   
  ##----------------------------------------     
  #We add the towers
  if len(towers)>0:     
    for ipos in range(len(towers[0])):
       if(0<=ietalisttower[ipos]<npoints-1 and 0<=iphilisttower[ipos]<npoints-1):
         toadd=[0,towers[0][ipos],0,0,0] #So images have [track pT,tower pT,track multiplicity,muon multiplicity,track charge] 
  #       toadd.extend([0,0,0,0,0,0])
         grid[ietalisttower[ipos],iphilisttower[ipos]]=grid[ietalisttower[ipos],iphilisttower[ipos]]+toadd
         nonzerogrid[ietalisttower[ipos],iphilisttower[ipos]]=1.
	 
	##---------------------------------------- 
	#So images have [track pT,tower pT,track multiplicity,muon multiplicity,track charge] 
	 
  test_output=np.nonzero(nonzerogrid) #Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.
  test_output2=grid[test_output[0],test_output[1]].tolist() #Elements that correspond to nonzero entries
  test_output3=np.transpose([test_output[0],test_output[1]]).tolist()
  test_output4=[list(a) for a in zip(test_output3,test_output2)] # We get the list with the pixel location and the pixel values only containing the nonzero entries
#   print('---'*20)
#   print('test_output4=',test_output4)

  ##----------------------------------------
  #We ask some treshold for the total pT fraction to keep the image when some constituents fall outside of the range for (eta,phi)
  sum=np.sum(grid)
  if sum<0.95:
    print('Error! image intensity below threshold!',sum)
#    print(ietalisttrack)
#    print(ietalisttower)
#    print(tracks[2])
#    print(iphilisttrack)
#    print(towers[2])
#    print(iphilisttower)


  final_image=test_output4

  return final_image


#-------------------------------------------------------------------------------------------------------------
#/////////////////////    MAIN FUNCTION THAT CALLS THE PREVIOUS ONES    /////////////////////////////////////
#------------------------------------------------------------------------------------------------------------- 
def preprocess_color_image(towerarray,trackarray,DReta,DRphi,npoints,preprocess_label):
 
  preprocess_cmnd=preprocess_label.split('_')

#   print('preprocess_cmnd=',preprocess_cmnd)
  trackpTarray=trackarray[0]
  tracketaarray=trackarray[1]
  trackphiarray=trackarray[2]
  trackmuonarray=trackarray[3]
  trackchargearray=trackarray[4]
   
  towerpTarray=towerarray[0]
  toweretaarray=towerarray[1]
  towerphiarray=towerarray[2]
  towermuonarray=np.zeros(len(towerarray[0]))
  towerchargearray=np.zeros(len(towerarray[0]))
  
  if(len(trackphiarray)>0):
    refphi=trackphiarray[0]
  else:
    refphi=towerphiarray[0]
          
# make sure the phi values are each the correct branch 
#  print(refphi)
#  print(trackphiarray)
  trackphiarray1=np.dstack((trackphiarray-refphi,trackphiarray-refphi-2*np.pi,trackphiarray-refphi+2*np.pi,trackphiarray-refphi-4*np.pi,trackphiarray-refphi+4*np.pi))[0]
  indphi=np.transpose(np.argsort(np.abs(trackphiarray1),axis=1))[0]
  trackphiarray2=trackphiarray1[np.arange(len(trackphiarray1)),indphi]+refphi
#  print(trackphiarray2)

#  print(towerphiarray)  
  towerphiarray1=np.dstack((towerphiarray-refphi,towerphiarray-refphi-2*np.pi,towerphiarray-refphi+2*np.pi,towerphiarray-refphi-4*np.pi,towerphiarray-refphi+4*np.pi))[0]
  indphi=np.transpose(np.argsort(np.abs(towerphiarray1),axis=1))[0]
  towerphiarray2=towerphiarray1[np.arange(len(towerphiarray1)),indphi]+refphi
#  print(towerphiarray2)

  TrackSubjets=[trackpTarray,tracketaarray,trackphiarray2,trackmuonarray,trackchargearray]
  TowerSubjets=[towerpTarray,toweretaarray,towerphiarray2,towermuonarray,towerchargearray]
#       AllSubjets=[np.concatenate((TrackSubjets[0],TowerSubjets[0])),np.concatenate((TrackSubjets[1],TowerSubjets[1])),np.concatenate((TrackSubjets[2],TowerSubjets[2]))]

# begin preprocessing
  
  ##----------------------------------------
  # Shift
  eta_c, phi_c=center(TrackSubjets,TowerSubjets)  
  TrackSubjets=shift(TrackSubjets,eta_c,phi_c)
  TowerSubjets=shift(TowerSubjets,eta_c,phi_c)

  ##----------------------------------------
  if('rot' in preprocess_cmnd):
    tan_theta=principal_axis(TrackSubjets,TowerSubjets) 
    TrackSubjets=rotate(TrackSubjets,tan_theta)
    TowerSubjets=rotate(TowerSubjets,tan_theta)

  ##----------------------------------------
  TrackSubjets=np.transpose(TrackSubjets)
  TowerSubjets=np.transpose(TowerSubjets)
  
  ##----------------------------------------
  if('vflip' in preprocess_cmnd):
    TrackSubjets,TowerSubjets=ver_flip_const(TrackSubjets,TowerSubjets)  
  
  ##----------------------------------------
  if('hflip' in preprocess_cmnd):
    TrackSubjets,TowerSubjets=hor_flip_const(TrackSubjets,TowerSubjets)  


  return TrackSubjets,TowerSubjets

#------------------------------------------------------------------------------------------------------------- 
# Create the image (after preprocessing)
def Make_Image(TrackSubjets,TowerSubjets,DReta,DRphi,npoints,preprocess_label):
#   if ('makeImage' in preprocess_cmnd):
    preprocess_cmnd=preprocess_label.split('_')
    
    TrackSubjets = copy.deepcopy(TrackSubjets)
    TowerSubjets = copy.deepcopy(TowerSubjets)
    
    TrackSubjets=np.transpose(TrackSubjets)
    TowerSubjets=np.transpose(TowerSubjets)  
  
    if('norm' in preprocess_cmnd):
      
      TrackSubjets,TowerSubjets=normalize(TrackSubjets,TowerSubjets)
      
    raw_image=create_color_image(TrackSubjets,TowerSubjets,DReta,DRphi,npoints)
    

    return raw_image

  
#------------------------------------------------------------------------------------------------------------- 
# Expand the images (only the non-zero entries were saved)
def expand_array(images,npoints_row=None, npoints_col=None,ncolors=None,norm_factor=None):
# ARRAY MUST BE IN THE FORM [[[iimage,ipixel,jpixel],val],...]

  Nimages=len(images)

#  print('Number of images ',Nimages)
  img_rows, img_cols = int(npoints_row-1), int(npoints_col-1)
  expandedimages=np.zeros((Nimages,img_rows,img_cols,ncolors))
#   expandedimages=np.zeros((Nimages,img_rows,img_cols))

  for i in range(Nimages):
#    print(i,len(images[i]))
#     sum_before=0
#     sum_after=0
    for j in range(len(images[i])): #We loop over each nonzero pixel. images[i][j]=[[eta,phi],[track pT,tower pT,track multiplicity,muon multiplicity,track charge]]
#       print(i,j,images[i][j][1])  

#       We normalize the track and tower pT by some number
  
#       sum_before+=images[i][j][1][0]+images[i][j][1][1]
      images[i][j][1][0]=images[i][j][1][0]/norm_factor
      images[i][j][1][1]=images[i][j][1][1]/norm_factor
#       sum_after+=images[i][j][1][0]+images[i][j][1][1]
      
      #-------
      #Make gray scale images
      expandedimages[i,images[i][j][0][0],images[i][j][0][1]] = images[i][j][1][0]+images[i][j][1][1] 
       # images[i][j][1][0] is the track pT and images[i][j][1][1] the tower pT. So we are just adding those 2 entries to get a gray scale image
#       print('expandedimages[i,images[i][j][0][0],images[i][j][0][1]]=',expandedimages[i,images[i][j][0][0],images[i][j][0][1]])
      
      
      #For color images 
#       expandedimages[i,images[i][j][0][0],images[i][j][0][1]] = images[i][j][1]
      
#       
#     #Normalize
#     print('Image sum before normalizing = ',np.sum(expandedimages[i]))
#     expandedimages[i]=expandedimages[i]/np.sum(expandedimages[i])
#     print('Image sum after normalizing = ',np.sum(expandedimages[i]))
#       
      
#     print('Image sum before normalizing = ',sum_before)     
#     print('Image sum after normalizing = ',sum_after) 
#   
#   
#     print('----------------'*10) 
#     print('expandedimages[i]=',expandedimages[i])
  
#   expandedimages=expandedimages.reshape(Nimages,img_rows,img_cols,ncolors)
  expandedimages=expandedimages.reshape(Nimages,img_rows,img_cols)
  
#  np.put(startgrid,ind,val)


  return expandedimages,img_rows,img_cols
  
  
#------------------------------------------------------------------------------------------------------------- 
# Add the images to get the average jet image for all the events
def add_images(Image,img_rows,img_cols):
  print('Adding the images to get the average jet image for all the events ...')
  print('-----------'*10)
  N_images=len(Image)
#   print('Number of images= {}'.format(N_images))
#   print('-----------'*10)
  avg_im=np.zeros((img_rows,img_cols)) #create an array of zeros for the image
  for ijet in range(0,len(Image)):
    avg_im=avg_im+Image[ijet]
    #avg_im2=np.sum(Image[ijet])
  print('Average image = \n {}'.format(avg_im))
  print('-----------'*10)
#  print('Average image 2 = \n {}'.format(avg_im2))
  #We normalize the image
  Total_int=np.absolute(np.sum(avg_im))
  print('Total intensity of average image = \n {}'.format(Total_int))
  print('-----------'*10)
#  norm_im=avg_im/Total_int
  norm_im=avg_im/N_images
#   print('Normalized average image (by number of images) = \n {}'.format(norm_im))
  print('Normalized average image = \n {}'.format(norm_im))
  print('-----------'*10)
  norm_int=np.sum(norm_im)
  print('Total intensity of average image after normalizing (should be 1) = \n {}'.format(norm_int))
  return norm_im, N_images
  
 
##---------------------------------------------------------------------------------------------
# Plot the averaged image
def plot_avg_image(Image, type,Nimages,DReta,DRphi,img_rows=None,name=None,Images_dir=None):
  print('Plotting the averaged image ...')
  print('-----------'*10)  
   
#   imgplot = plt.imshow(Image[0], 'viridis')# , origin='upper', interpolation='none', vmin=0, vmax=0.5)  
  imgplot = plt.imshow(Image, 'gnuplot', extent=[-DReta, DReta,-DRphi, DRphi])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
#   imgplot = plt.imshow(Image[0])
#   plt.show()
  plt.xlabel('$\eta^{\prime\prime}$')
  plt.ylabel('$\phi^{\prime\prime}$')
  fig = plt.gcf()
  image_name=str(name)+'_avg_im_'+str(Nimages)+'_'+str(img_rows)+'_'+type+'.png'
  plt.savefig(Images_dir+image_name)
  print('Average image filename = {}'.format(Images_dir+image_name))


#------------------------------------------------------------------------------------------------------------- 
# Function to plot the histograms
def makeHist(out_dir,data,bins,plotname,title,xaxis,yaxis,type,Njets):
  myfig = plt.figure()
  ax1 = myfig.add_subplot(1, 1, 1)
  n, bins, patches = ax1.hist(data,bins,alpha=0.5)
  ax1.set_xlabel(str(xaxis))
  ax1.set_ylabel(str(yaxis))
  ax1.set_title('Histogram of '+str(title))
  ax1.grid(True)
  plot_FNAME = 'Hist_'+str(plotname)+'_'+type+'_'+str(Njets)+'.png'
  print('------------'*10)
  print('Hist plot = ',out_dir+'/'+plot_FNAME)
  print('------------'*10)
  plt.savefig(out_dir+'/'+plot_FNAME)

def make2DHist(out_dir,data1,data2,bins,plotname,title,xaxis,yaxis,type,Njets,xmin=None,xmax=None,ymin=None, ymax=None):
  myfig = plt.figure()
  ax1 = myfig.add_subplot(1, 1, 1)
#   fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True,tight_layout=True)
#   n, bins, patches = ax1.hist2d(data1,data2,bins)
  ax1.hist2d(data1,data2,bins,range=[[xmin, xmax], [ymin, ymax]])
  ax1.set_xlabel(str(xaxis))
  ax1.set_ylabel(str(yaxis))
  ax1.set_title('Histogram of '+str(title))
  ax1.grid(True)
  plot_FNAME = 'Hist_'+str(plotname)+'_'+type+'_'+str(Njets)+'.png'
  print('------------'*10)
  print('Hist plot = ',out_dir+'/'+plot_FNAME)
  print('------------'*10)
  plt.savefig(out_dir+'/'+plot_FNAME)

#-------------------------------------------------------------------------------------------------------------- 












