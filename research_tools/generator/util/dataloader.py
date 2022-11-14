import torch
import time


class ValDataPrefetcher:

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = ValDataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_img, self.next_target, self.next_img_info, self.next_img_id = next(self.loader)
        except StopIteration:
            self.next_img = None
            self.next_target = None
            self.next_img_info = None
            self.next_img_id = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.next_img
        target = self.next_target
        img_info = self.next_img_info
        img_id = self.next_img_id
        if img is not None:
            self.record_stream(img)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return img, target, img_info, img_id

    def _input_cuda_for_image(self):
        self.next_img = self.next_img.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
