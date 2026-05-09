import { Routes } from '@angular/router';
import { LiveView } from '../../pages/live_view/liveView';
import { PhotoUpload } from '../../pages/photo_upload/photoUpload';
export const routes: Routes = [
    {
        path: '',
        redirectTo: 'live-view',
        pathMatch: 'full'
    },
    {
        path: 'live-view',
        component: LiveView
    },
    {
        path: 'photo-upload',
        component: PhotoUpload
    },
    {
        path: '**',
        redirectTo: 'live-view'
    }
];
