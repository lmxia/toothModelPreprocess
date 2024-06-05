import tornado.ioloop
import tornado.web
import torch
import trimesh
import os
from obs import ObsClient
import json
import numpy as np
from train import TeethAlignmentModel


class InferenceHandler(tornado.web.RequestHandler):
    def initialize(self, model, obs_client, bucket_name, standard_cloud):
        self.model = model
        self.obs_client = obs_client
        self.bucket_name = bucket_name
        self.standard_cloud = standard_cloud

    async def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            object_key = data['object_key']
            file_path = f'/tmp/{object_key}'

            # Download file from OBS
            self.obs_client.getObject(self.bucket_name, object_key, file_path)
            mesh = trimesh.load(file_path)
            os.remove(file_path)

            # Process point cloud
            points = mesh.vertices
            points = points - np.mean(points, axis=0)
            points = points / np.max(np.linalg.norm(points, axis=1))

            if len(points) != 24000:
                if len(points) > 24000:
                    points = self.downsample(points, 24000)
                else:
                    points = self.upsample(points, 24000)

            points = torch.tensor(points, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)  # Reshape to match model input

            # Model inference
            self.model.eval()
            with torch.no_grad():
                _, transformed_points = self.model(points, self.standard_cloud.unsqueeze(0).permute(0, 2, 1))
                transformed_points = transformed_points.squeeze().permute(1, 0).numpy()

            response = {
                "transformed_points": transformed_points.tolist()
            }
            self.write(json.dumps(response))

        except Exception as e:
            self.write(json.dumps({"error": str(e)}))

    def downsample(self, points, num_points):
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        return points[indices]

    def upsample(self, points, num_points):
        indices = np.random.choice(points.shape[0], num_points, replace=True)
        return points[indices]


def make_app(model, obs_client, bucket_name, standard_cloud):
    return tornado.web.Application([
        (r"/infer", InferenceHandler, dict(model=model, obs_client=obs_client, bucket_name=bucket_name, standard_cloud=standard_cloud)),
    ])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Teeth alignment inference server')
    parser.add_argument('--port', type=int, default=8888, help='Port to listen on')
    parser.add_argument('--obs_access_key', type=str, required=True, help='OBS access key')
    parser.add_argument('--obs_secret_key', type=str, required=True, help='OBS secret key')
    parser.add_argument('--obs_endpoint', type=str, required=True, help='OBS endpoint')
    parser.add_argument('--bucket_name', type=str, required=True, help='OBS bucket name')
    parser.add_argument('--model_path', type=str, default='/home/teeth_alignment_model.pth', help='Path to the trained model')
    parser.add_argument('--standard_model_path', type=str, required=True, help='Path to the standard model STL file')

    args = parser.parse_args()

    # Load the model
    model = TeethAlignmentModel()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Load standard model point cloud
    standard_mesh = trimesh.load(args.standard_model_path)
    standard_cloud = standard_mesh.vertices
    standard_cloud = standard_cloud - np.mean(standard_cloud, axis=0)
    standard_cloud = standard_cloud / np.max(np.linalg.norm(standard_cloud, axis=1))
    if len(standard_cloud) != 24000:
        if len(standard_cloud) > 24000:
            standard_cloud = np.random.choice(standard_cloud, 24000, replace=False)
        else:
            standard_cloud = np.random.choice(standard_cloud, 24000, replace=True)
    standard_cloud = torch.tensor(standard_cloud, dtype=torch.float32)

    # Initialize OBS client
    obs_client = ObsClient(
        access_key_id=args.obs_access_key,
        secret_access_key=args.obs_secret_key,
        server=args.obs_endpoint
    )

    app = make_app(model, obs_client, args.bucket_name, standard_cloud)
    app.listen(args.port)
    print(f"Server running on port {args.port}")
    tornado.ioloop.IOLoop.current().start()
