import numpy as np
import tornado.ioloop
import tornado.web
import torch
import os
from obs import ObsClient
import json
from train import TeethAlignmentModel
from utils import gen_util as gu
from chamferdist import ChamferDistance

class InferenceHandler(tornado.web.RequestHandler):
    def initialize(self, model, obs_client, bucket_name, standard_cloud, target_vector):
        self.model = model
        self.obs_client = obs_client
        self.bucket_name = bucket_name
        self.standard_cloud = standard_cloud
        self.target_vector  = target_vector

    async def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            object_key = data['object_key']
            file_path = f'/tmp/{object_key}'
            obj_name = object_key.split(".")[0]
            transformed_key = f"{obj_name}.obj"
            transformed_path = f'/tmp/{transformed_key}'

            # Download file from OBS
            self.obs_client.getObject(self.bucket_name, object_key, file_path)
            vertices = gu.load_and_sample_mesh(file_path)
            os.remove(file_path)

            points = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)  # Reshape to match model input
            chamfer_dist = ChamferDistance()
            # Model inference
            self.model.eval()
            with torch.no_grad():
                rot, trans = self.model(points, self.standard_cloud.unsqueeze(0))
                source_transformed = gu.apply_transform(points, rot, trans)
                loss = gu.compute_loss(chamfer_dist, source_transformed, self.standard_cloud.unsqueeze(0), self.target_vector)
                transformed_points = source_transformed.squeeze().numpy()
                gu.logger.info(f"total loss is {loss}")
            #
            # mesh.vertices = transformed_points
            # mesh.export(transformed_path)

            with open(transformed_path, 'w') as file:
                for point in transformed_points:
                    file.write(f"v {point[0]} {point[1]} {point[2]}\n")

            # Upload transformed OBJ to OBS
            self.obs_client.putFile(self.bucket_name, transformed_key, transformed_path)
            os.remove(transformed_path)

            response = {
                "transformed_key": transformed_key
            }
            self.write(json.dumps(response))
        except Exception as e:
            self.write(json.dumps({"error": str(e)}))


def make_app(model, obs_client, bucket_name, standard_cloud, target_vector):
    return tornado.web.Application([
        (r"/infer", InferenceHandler, dict(model=model, obs_client=obs_client, bucket_name=bucket_name,
                                           standard_cloud=standard_cloud, target_vector=target_vector)),
    ])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Teeth alignment inference server')
    parser.add_argument('--port', type=int, default=8888, help='Port to listen on')
    parser.add_argument('--obs_access_key', type=str, required=True, help='OBS access key')
    parser.add_argument('--obs_secret_key', type=str, required=True, help='OBS secret key')
    parser.add_argument('--obs_endpoint', type=str, default="obs.cn-east-3.myhuaweicloud.com", help='OBS endpoint')
    parser.add_argument('--bucket_name', type=str, default="teeth-label-dev", help='OBS bucket name')
    parser.add_argument('--model_path', type=str, default='/data/teeth_alignment_model.pth', help='Path to the trained model')
    parser.add_argument('--standard_model_path', type=str, default='/data/xia.stl', help='Path to the standard model STL file')

    args = parser.parse_args()

    # Load the model
    model = TeethAlignmentModel()

    _, state_dict, _, _ = gu.load_checkpoint(args.model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Load standard model point cloud
    vertices = gu.load_and_sample_mesh(args.standard_model_path)
    standard_cloud = torch.tensor(vertices, dtype=torch.float32)

    target_points_batch = np.expand_dims(vertices, axis=0)
    target_vector = gu.compute_centroid_direction_vector(target_points_batch[:, :, :3])[0]
    target_vector = np.tile(target_vector, (1, 1))
    # Initialize OBS client
    obs_client = ObsClient(
        access_key_id=args.obs_access_key,
        secret_access_key=args.obs_secret_key,
        server=args.obs_endpoint
    )

    app = make_app(model, obs_client, args.bucket_name, standard_cloud, target_vector)
    app.listen(args.port)
    print(f"Server running on port {args.port}")
    tornado.ioloop.IOLoop.current().start()
